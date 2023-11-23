import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    MCAcquisitionObjective,
    qExpectedImprovement,
    qMultiStepLookahead, PosteriorMean, MCAcquisitionFunction  # Re-written to allow passing inner samplers
)
from botorch.acquisition.multi_step_lookahead import make_best_f, _construct_sample_weights, _construct_inner_samplers
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points, match_batch_shape
from torch import Tensor
from torch.nn import ModuleList

TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]


class qEIMLMCOneStep:
    r"""Class for one-step lookahead q-EI with MLMC (2-EI) by default"""

    def __init__(self, model, bounds, num_restarts, raw_samples, batch_sizes=[2]):
        r"""
        Args:
            bounds: bounds of objective function
            model: a fitted single-outcome model
            num_restarts: number of restarts for LBFGS
            raw_samples: number of raw samples for LBFGS
            batch_sizes: array for q of q-EI, default is [2] (2-EI)
        """
        self.model = model
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_sizes = batch_sizes

    @staticmethod
    def _create_sampler(num_samples, seed):
        r"""Initialize sampler with a seed for coupling"""
        return IIDNormalSampler(sample_shape=torch.Size([num_samples]), seed=seed)

    def sample_candidate(self, l, dl, num_samples, match=None):
        r"""Generate new observations with MLMC
        Args:
            l: index level of estimation (starting from 0)
            dl: starting level (l <- l + dl)
            num_samples: number of outer samples
            match: if True matching the optimizer generated at level l with level l-1,
                   so that MLMC tracks the same optimizer among levels;
                   No need to match at level 0
        Returns:
            Three elements tuple
            new_candidate: next candidate point by MLMC (optimizer)
            new_value: corresponding value of objective by MLMC (optimum)
            match_candidate: candidate used for matching in the next level
        """

        # number of inner samples
        M = 2 ** (l + dl)

        # fixed seed of base samples for coupling
        # No need to coupling for level 0 (single level estimator instead of increments)
        seed_out, seed_in = (torch.randint(0, 10000000, (1,)).item() for _ in range(2))

        samplers = [self._create_sampler(num_samples, seed_out)]
        inner_mc_samplers = [None, self._create_sampler(M, seed_in)]

        if l == 0:
            new_candidate, new_value = self.get_candidates(samplers, inner_mc_samplers)
            match_candidate = new_candidate
        else:
            if match is not None:
                new_candidate_f, new_value_f = self.get_candidates(samplers, inner_mc_samplers,
                                                                   return_best=False)
                new_candidate_c, new_value_c = self.get_candidates(samplers, inner_mc_samplers,
                                                                   antithetic=True,
                                                                   return_best=False)
                diff_c = torch.argmin(torch.norm(match - new_candidate_c, dim=2))
                diff_f = torch.argmin(torch.norm(new_candidate_f - new_candidate_c[diff_c], dim=2))
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_f[diff_f]
            else:
                new_candidate_f, new_value_f = self.get_candidates(samplers, inner_mc_samplers)
                new_candidate_c, new_value_c = self.get_candidates(samplers, inner_mc_samplers)
                new_candidate = new_candidate_f - new_candidate_c
                new_value = new_value_f - new_value_c
                match_candidate = new_candidate_c

        return new_candidate, new_value, match_candidate

    def get_candidates(self, samplers, inner_mc_samplers, antithetic=False, return_best=True):
        r"""Generate the next observation of single one-step lookahead EI.
        Args:
            samplers: samplers for outer MC
            inner_mc_samplers: samplers for inner MC
            antithetic: whether to compute antithetic estimator
            return_best: whether to return the best result of objective function optimization
        Returns:
            new_candidate: next candidate point (optimizer)
            new_value: corresponding value of objective (optimum)
        """

        # initialize the acquisition function for the zero and first steps
        valfunc_cls = [ExpectedImprovement, qExpectedImprovement if antithetic is False else qExpectedImprovementAnt]
        valfunc_argfacs = [make_best_f, make_best_f]

        oneqEI = CustomqMultiStepLookahead(
            model=self.model,
            batch_sizes=self.batch_sizes,
            num_fantasies=None,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            inner_mc_samplers=inner_mc_samplers
        )

        q_prime = oneqEI.get_augmented_q_batch_size(1)
        new_candidate, new_value = optimize_acqf(
            acq_function=oneqEI,
            bounds=self.bounds,
            q=q_prime,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            return_best_only=return_best
        )

        return new_candidate, new_value


class CustomqMultiStepLookahead(qMultiStepLookahead):
    r"""Inherited from qMultiStepLookahead allowing passing inner samplers"""

    def __init__(
            self,
            model: Model,
            batch_sizes: List[int],
            num_fantasies: Optional[List[int]] = None,
            samplers: Optional[List[MCSampler]] = None,
            # Modification to allow passing in inner samplers directly
            inner_mc_samplers: Optional[List[Optional[MCSampler]]] = None,
            valfunc_cls: Optional[List[Optional[Type[AcquisitionFunction]]]] = None,
            valfunc_argfacs: Optional[List[Optional[TAcqfArgConstructor]]] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            inner_mc_samples: Optional[List[int]] = None,
            X_pending: Optional[Tensor] = None,
            collapse_fantasy_base_samples: bool = True
    ) -> None:
        r"""rewrite the initializer of qMultistepLookahead by allowing passing inner samplers"""

        if objective is not None and not isinstance(objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "`qMultiStepLookahead` got a non-MC `objective`. This is not supported."
                " Use `posterior_transform` and `objective=None` instead."
            )

        super(MCAcquisitionFunction, self).__init__(model=model)
        self.batch_sizes = batch_sizes
        if not ((num_fantasies is None) ^ (samplers is None)):
            raise UnsupportedError(
                "qMultiStepLookahead requires exactly one of `num_fantasies` or "
                "`samplers` as arguments."
            )
        if samplers is None:
            # If collapse_fantasy_base_samples is False, the `batch_range_override`
            # is set on the samplers during the forward call.
            samplers: List[MCSampler] = [
                SobolQMCNormalSampler(sample_shape=torch.Size([nf]))
                for nf in num_fantasies
            ]
        else:
            num_fantasies = [sampler.sample_shape[0] for sampler in samplers]
        self.num_fantasies = num_fantasies
        # By default do not use stage values and use PosteriorMean as terminal value
        # function (= multi-step KG)
        if valfunc_cls is None:
            valfunc_cls = [None for _ in num_fantasies] + [PosteriorMean]
        if inner_mc_samplers is not None:
            inner_samplers = inner_mc_samplers
        else:
            if inner_mc_samples is None:
                inner_mc_samples = [None] * (1 + len(num_fantasies))
            inner_samplers = _construct_inner_samplers(
                batch_sizes=batch_sizes,
                valfunc_cls=valfunc_cls,
                objective=objective,
                inner_mc_samples=inner_mc_samples,
            )
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


class qExpectedImprovementAnt(qExpectedImprovement):
    r"""qExpectedImprovement antithetic estimation."""

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            X_pending: Optional[Tensor] = None,
            **kwargs: Any,
    ) -> None:
        r"""
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
        r"""Evaluate qExpectedImprovement antithetic on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of antithetic of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)

        # separate the samples and compute each estimation for antithetic approach
        obj_1 = obj[1::2]
        obj_1 = (obj_1 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei_1 = obj_1.max(dim=-1)[0].mean(dim=0)
        obj_2 = obj[::2]
        obj_2 = (obj_2 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei_2 = obj_2.max(dim=-1)[0].mean(dim=0)
        return (q_ei_1 + q_ei_2) / 2


class qEIMLMCTwoStep:
    def __init__(self, model, bounds, num_restarts, raw_samples):
        self.model = model
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    @staticmethod
    def _create_sampler(num_samples, base_samples):
        sampler = IIDNormalSampler(sample_shape=torch.Size([num_samples]), resample=False)
        sampler.base_samples = base_samples
        return sampler

    def get_candidates(self, samplers, fc):
        twoEI = TwoStepIncAntEI(
            model=self.model,
            samplers=samplers,
            bounds=self.bounds,
            fc=fc
        )
        q = twoEI.get_augmented_q_batch_size(1)
        new_candidate, _ = optimize_acqf(
            acq_function=twoEI,
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples
        )
        return new_candidate

    def sample_candidate(self, l, dl, num_samples, **kwargs):
        M = 2 ** (l + dl)

        base_samples1 = torch.randn(num_samples)
        base_samples2 = torch.randn(M)

        sampler1 = self._create_sampler(num_samples, base_samples1)

        if l == 0:
            sampler2 = self._create_sampler(M, base_samples2)
            new_candidate = self.get_candidates([sampler1, sampler2], fc=0)
        else:
            sampler2f = self._create_sampler(M, base_samples2)
            new_candidate_f = self.get_candidates([sampler1, sampler2f], fc=0)

            sampler2c1 = self._create_sampler(M // 2, base_samples2[1::2])
            sampler2c2 = self._create_sampler(M // 2, base_samples2[::2])
            new_candidate_c = self.get_candidates([sampler1, sampler2c1, sampler2c2], fc=1)

            new_candidate = new_candidate_f - new_candidate_c

        return new_candidate


class TwoStepIncAntEI(qMultiStepLookahead):

    def __init__(
            self,
            model: Model,
            bounds: Tensor,
            samplers: List[MCSampler],
            fc: int,
            **kwargs
    ) -> None:
        self.k = 2
        batch_sizes = [1, 1]
        super().__init__(model=model,
                         batch_sizes=batch_sizes,
                         samplers=samplers,
                         **kwargs)
        if fc == 1:
            num_fantasies = [samplers[i].sample_shape[0] for i in range(len(samplers) - 1)]
            self.num_fantasies = num_fantasies
        self.bounds = bounds
        self.fc = fc

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xs = self.get_multi_step_tree_input_representation(X)

        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        X0 = Xs[0]
        sample_weights = torch.ones(*X0.shape[:-2], device=X.device, dtype=X.dtype)
        best_f0 = self.model.train_targets.max()
        stage_val_func0 = ExpectedImprovement(model=self.model, best_f=best_f0)
        stage_val0 = stage_val_func0(X=X0)
        running_val = stage_val0

        fantasy_model1 = self.model.fantasize(
            X=X0, sampler=self.samplers[0], observation_noise=False, propagate_grads=True
        )

        sample_weights = _construct_sample_weights(
            prev_weights=sample_weights, sampler=self.samplers[0]
        )
        if self.fc == 0:

            X1 = Xs[1]
            best_f1 = fantasy_model1.train_targets.max()
            stage_val_func1 = ExpectedImprovement(model=fantasy_model1, best_f=best_f1)
            stage_val1 = stage_val_func1(X=X1)
            running_val = running_val + stage_val1

            X2 = Xs[2]
            fantasy_model2 = fantasy_model1.fantasize(
                X=X1, sampler=self.samplers[1], observation_noise=False, propagate_grads=True
            )
            sample_weights = _construct_sample_weights(
                prev_weights=sample_weights, sampler=self.samplers[1]
            )

            best_f2 = fantasy_model2.train_targets.max()
            stage_val_func2 = ExpectedImprovement(model=fantasy_model2, best_f=best_f2)
            stage_val2 = stage_val_func2(X=X2)
            running_val = running_val + stage_val2

            batch_shape = running_val.shape[self.k:]
            sample_weights = sample_weights.expand(running_val.shape)
            return (running_val * sample_weights).view(-1, *batch_shape).sum(dim=0)
        else:
            X11 = Xs[1]
            best_f1 = fantasy_model1.train_targets.max()
            stage_val_func11 = ExpectedImprovement(model=fantasy_model1, best_f=best_f1)
            stage_val11 = stage_val_func11(X=X11)
            running_val1 = running_val + stage_val11

            X21 = Xs[2]
            fantasy_model21 = fantasy_model1.fantasize(
                X=X11, sampler=self.samplers[1], observation_noise=False, propagate_grads=True
            )

            best_f21 = fantasy_model21.train_targets.max()
            stage_val_func21 = ExpectedImprovement(model=fantasy_model21, best_f=best_f21)
            stage_val21 = stage_val_func21(X=X21)
            running_val1 = running_val1 + stage_val21

            batch_shape1 = running_val1.shape[self.k:]
            sample_weights1 = sample_weights.expand(running_val1.shape)

            X12 = Xs[3]
            stage_val_func12 = ExpectedImprovement(model=fantasy_model1, best_f=best_f1)
            stage_val12 = stage_val_func12(X=X12)
            running_val2 = running_val + stage_val12

            X22 = Xs[4]
            fantasy_model22 = fantasy_model1.fantasize(
                X=X12, sampler=self.samplers[2], observation_noise=False, propagate_grads=True
            )

            best_f22 = fantasy_model22.train_targets.max()
            stage_val_func22 = ExpectedImprovement(model=fantasy_model22, best_f=best_f22)
            stage_val22 = stage_val_func22(X=X22)
            running_val2 = running_val2 + stage_val22

            batch_shape2 = running_val2.shape[self.k:]
            sample_weights2 = sample_weights.expand(running_val2.shape)

            return ((running_val1 * sample_weights1).view(-1, *batch_shape1).sum(dim=0) + \
                    (running_val2 * sample_weights2).view(-1, *batch_shape2).sum(dim=0)) / 2

    @property
    def _num_auxiliary(self) -> int:
        r"""Number of auxiliary variables in the q-batch dimension.

        Returns:
             `q_aux` s.t. `q + q_aux = augmented_q_batch_size`
        """
        if self.fc == 1:
            return 2 * super()._num_auxiliary
        return super()._num_auxiliary

    def get_multi_step_tree_input_representation(self, X: Tensor) -> List[Tensor]:
        r"""Get the multi-step tree representation of X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.

        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        if self.fc == 1:
            sizes += sizes[1:]
            shapes += shapes[1:]
        Xsplit = torch.split(X, sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)

        return [X0] + Xother
