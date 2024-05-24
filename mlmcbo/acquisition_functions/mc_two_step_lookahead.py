from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from botorch import settings
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    qKnowledgeGradient,
    MCAcquisitionObjective,
    qExpectedImprovement,
    qMultiStepLookahead
)
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_step_lookahead import make_best_f
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform, concatenate_pending_points
from torch import Tensor
from torch.nn import ModuleList

TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]


class ExpectedImprovementTwoStepLookahead(qKnowledgeGradient):
    r"""two-step lookahead expected improvement
    implemented in a one-shot fashion"""

    def __int__(
            self,
            model: Model,
            num_fantasies: Optional[int] = None,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            inner_sampler: Optional[MCSampler] = None,
            X_pending: Optional[Tensor] = None,
            current_value: Optional[Tensor] = None
    ) -> None:
        super().__init__(
            model,
            num_fantasies,
            sampler,
            objective,
            inner_sampler,
            X_pending,
            current_value
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate one step-lookahead qEI on the candidate set 'X'

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`.
            For t-batch b, the one-step lookahead EI value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
        """
        X_actual, X_fantasies = _split_fantasy_points(
            X=X, n_f=self.num_fantasies
        )

        current_value = (
            self.current_value if self.current_value else self.model.train_targets.max()
        )

        ei = ExpectedImprovement(model=self.model, best_f=current_value)
        zero_step_ei = ei(X_actual)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=False
        )

        best_f = fantasy_model.train_targets.max(dim=-1)[0]

        if not self.inner_sampler:
            one_step_ei = ExpectedImprovement(model=fantasy_model, best_f=best_f)
        else:
            one_step_ei = qExpectedImprovement(model=fantasy_model,
                                               sampler=self.inner_sampler,
                                               best_f=best_f)

        with settings.propagate_grads(True):
            values = one_step_ei(X=X_fantasies)

        one_step_ei_avg = values.mean(dim=0)

        return zero_step_ei + one_step_ei_avg


class qExpectedImprovementAnt(qExpectedImprovement):
    r"""qExpectedImprovement antithetic estimation"""

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """

        super().__init__(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)

        obj_1 = obj[1::2]
        obj_1 = (obj_1 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei_1 = obj_1.max(dim=-1)[0].mean(dim=0)
        obj_2 = obj[::2]
        obj_2 = (obj_2 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei_2 = obj_2.max(dim=-1)[0].mean(dim=0)
        return (q_ei_1 + q_ei_2) / 2


class qExpectedImprovementTwoStepLookahead(qMultiStepLookahead):

    def __init__(
            self,
            model: Model,
            batch_sizes: List[int],
            num_fantasies: Optional[List[int]] = None,
            samplers: Optional[List[MCSampler]] = None,
            valfunc_cls: Optional[List[Optional[Type[AcquisitionFunction]]]] = [ExpectedImprovement,
                                                                                qExpectedImprovement],
            valfunc_argfacs: Optional[List[Optional[TAcqfArgConstructor]]] = [make_best_f, make_best_f],
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            inner_mc_samples: Optional[List[int]] = None,
            X_pending: Optional[Tensor] = None,
            collapse_fantasy_base_samples: bool = True,
            antithetic_variates=False,
            num_samples=512,
            num_samples_inner=512
    ) -> None:
        inner_samplers = [None, IIDNormalSampler(sample_shape=torch.Size([num_samples]), resample=False)]

        if objective is not None and not isinstance(objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "`qMultiStepLookahead` got a non-MC `objective`. This is not supported."
                " Use `posterior_transform` and `objective=None` instead."
            )

        super(MCAcquisitionFunction, self).__init__(model=model)
        self.batch_sizes = batch_sizes
        if samplers is None:
            samplers = [IIDNormalSampler(sample_shape=torch.Size([num_samples]), resample=False)]
        if not ((num_fantasies is None) ^ (samplers is None)):
            raise UnsupportedError(
                "qMultiStepLookahead requires exactly one of `num_fantasies` or "
                "`samplers` as arguments."
            )
        num_fantasies = [sampler.sample_shape[0] for sampler in samplers]
        self.num_fantasies = num_fantasies
        if antithetic_variates:
            valfunc_cls = [ExpectedImprovement, qExpectedImprovementAnt]
        if inner_mc_samples is None:
            inner_mc_samples = [None] * (1 + len(num_fantasies))
        if valfunc_argfacs is None:
            valfunc_argfacs = [None] * (1 + len(batch_sizes))

        self.objective = objective
        self.posterior_transform = posterior_transform
        self.set_X_pending(X_pending)
        self.samplers = ModuleList(samplers)
        self.inner_samplers = ModuleList(inner_samplers)
        self._valfunc_cls = valfunc_cls
        self._valfunc_argfacs = valfunc_argfacs
        self._collapse_fantasy_base_samples = collapse_fantasy_base_samples


def _split_fantasy_points(X: Tensor, n_f: int) -> Tuple[Tensor, Tensor]:
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.
    # b x num_fantasies x d --> num_fantasies x b x d
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    # num_fantasies x b x 1 x d
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies
