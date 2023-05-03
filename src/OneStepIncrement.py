import torch
from typing import Optional, Tuple, Union, Any

from botorch import settings
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, qKnowledgeGradient, \
    MCAcquisitionObjective, qExpectedImprovement, MCAcquisitionFunction, OneShotAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform, concatenate_pending_points
from torch import Tensor


class OneStepIncEI(qKnowledgeGradient):
    r"""increment of one-step lookahead expected improvement
    implemented in a one-shot fashion"""

    def __init__(
            self,
            model: Model,
            bounds: Tensor,
            num_restarts: int,
            raw_samples: int,
            fc: int,
            num_fantasies: Optional[int] = None,
            sampler: Optional[MCSampler] = None,
            inner_sampler: Optional[MCSampler] = None,
            current_value: Optional[Tensor] = None,
    ) -> None:
        # if sampler is None:
        #     if num_fantasies is None:
        #         raise ValueError(
        #             "Must specify `num_fantasies` if no `sampler` is provided."
        #         )
        #     # base samples should be fixed for joint optimization over X, X_fantasies
        #     sampler = SobolQMCNormalSampler(
        #         num_samples=num_fantasies, resample=False, collapse_batch_dims=True
        #     )
        # elif num_fantasies is not None:
        #     if sampler.sample_shape != torch.Size([num_fantasies]):
        #         raise ValueError(
        #             f"The sampler shape must match num_fantasies={num_fantasies}."
        #         )
        # else:
        #     num_fantasies = sampler.sample_shape[0]
        super().__init__(model=model,
                         sampler=sampler,
                         num_fantasies=num_fantasies,
                         inner_sampler=inner_sampler)

        # self.num_fantasies = num_fantasies
        # self.inner_sampler = inner_sampler
        self.fc = fc
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.current_value = current_value

    # @concatenate_pending_points
    # @t_batch_mode_transform()
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
        # if self.X_pending is not None:
        #     X_actual = torch.cat(
        #         [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
        #     )

        # from botorch.generation.gen import gen_candidates_scipy

        # optimize the inner problem
        # from botorch.optim.initializers import gen_value_function_initial_conditions

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=False
        )

        best_fan = fantasy_model.train_targets.max(dim=-1)[0]
        #
        if self.fc == 0:
            one_step_ei = ExpectedImprovementInc(model=fantasy_model,
                                                 sampler=self.inner_sampler,
                                                 best_f=best_fan,
                                                 fc=self.fc)

            with settings.propagate_grads(True):
                values = one_step_ei(X=X_fantasies)

            one_step_ei_avg = values.mean(dim=0)
            return zero_step_ei + one_step_ei_avg
        elif self.fc == 1:
            # one_step_ei_c1 = qExpectedImprovement(model=fantasy_model,
            #                                       sampler=self.inner_sampler,
            #                                       best_f=best_fan,
            #                                       )
            # one_step_ei_f = ExpectedImprovementInc(model=fantasy_model,
            #                                      sampler=self.inner_sampler,
            #                                      best_f=best_fan,
            #                                      fc=0)
            one_step_ei = ExpectedImprovementInc(model=fantasy_model,
                                                 sampler=self.inner_sampler,
                                                 best_f=best_fan,
                                                 fc=1)

        # one_step_ei_c2 = ExpectedImprovementInc(model=fantasy_model,
        #                                         sampler=self.inner_sampler,
        #                                         best_f=best_fan,
        #                                         fc=2)

            with settings.propagate_grads(True):
                # values_f = one_step_ei_f(X=X_fantasies)
                values = one_step_ei(X=X_fantasies)
                # values_c2 = one_step_ei(X=X_fantasies)[1]
            # one_step_ei_avg = (values_c1.mean(dim=0) + values_c2.mean(dim=0)) / 2
            # one_step_ei_avg_f = values_f.mean(dim=0)
            one_step_ei_avg = values.mean(dim=0)
            return zero_step_ei + one_step_ei_avg

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : - self.num_fantasies, :]


class OneStepIncAntEI(qKnowledgeGradient):
    r"""antithetic increment of one-step lookahead expected improvement
    implemented in a one-shot fashion"""

    def __init__(
            self,
            model: Model,
            bounds: Tensor,
            num_restarts: int,
            raw_samples: int,
            fc: int,
            num_fantasies: Optional[int] = None,
            sampler: Optional[MCSampler] = None,
            inner_sampler: Optional[MCSampler] = None,
            current_value: Optional[Tensor] = None,
    ) -> None:
        # if sampler is None:
        #     if num_fantasies is None:
        #         raise ValueError(
        #             "Must specify `num_fantasies` if no `sampler` is provided."
        #         )
        #     # base samples should be fixed for joint optimization over X, X_fantasies
        #     sampler = SobolQMCNormalSampler(
        #         num_samples=num_fantasies, resample=False, collapse_batch_dims=True
        #     )
        # elif num_fantasies is not None:
        #     if sampler.sample_shape != torch.Size([num_fantasies]):
        #         raise ValueError(
        #             f"The sampler shape must match num_fantasies={num_fantasies}."
        #         )
        # else:
        #     num_fantasies = sampler.sample_shape[0]
        super().__init__(model=model,
                         sampler=sampler,
                         num_fantasies=num_fantasies,
                         inner_sampler=inner_sampler)

        # self.num_fantasies = num_fantasies
        # self.inner_sampler = inner_sampler
        self.fc = fc
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.current_value = current_value

    @concatenate_pending_points
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
        # if self.X_pending is not None:
        #     X_actual = torch.cat(
        #         [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
        #     )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=False
        )

        best_fan = fantasy_model.train_targets.max(dim=-1)[0]

        if self.fc == 0:
            one_step_ei = ExpectedImprovementInc(model=fantasy_model,
                                                 sampler=self.inner_sampler,
                                                 best_f=best_fan,
                                                 fc=self.fc)

            with settings.propagate_grads(True):
                values = one_step_ei(X=X_fantasies)

            one_step_ei_avg = values.mean(dim=0)
        elif self.fc == 1:
            seed = self.inner_sampler.seed
            one_step_ei_c1 = ExpectedImprovementInc(model=fantasy_model,
                                                    sampler=self.inner_sampler,
                                                    best_f=best_fan,
                                                    fc=1)
            self.inner_sampler.seed = seed
            one_step_ei_c2 = ExpectedImprovementInc(model=fantasy_model,
                                                    sampler=self.inner_sampler,
                                                    best_f=best_fan,
                                                    fc=2)

            with settings.propagate_grads(True):
                values_c1 = one_step_ei_c1(X=X_fantasies)
                values_c2 = one_step_ei_c2(X=X_fantasies)

            one_step_ei_avg = (values_c1.mean(dim=0) + values_c2.mean(dim=0)) / 2
            # one_step_ei_avg = values_c1.mean(dim=0)

        return zero_step_ei + one_step_ei_avg

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : - self.num_fantasies, :]


def _split_fantasy_points(X: Tensor, n_f: int) -> Tuple[Tensor, Tensor]:
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

class ExpectedImprovementInc(qExpectedImprovement):

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            fc: int,
            sampler: Optional[MCSampler] = None,
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
            sampler=sampler,
            best_f=best_f
        )
        self.fc = fc

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
        # if self.fc == 0:
        #     obj = self.objective(samples, X=X)
        #     obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        #     q_ei = obj.max(dim=-1)[0].mean(dim=0)
        #     return q_ei
        # elif self.fc == 1:
        #     obj_c1 = self.objective(samples[1::2], X=X)
        #     obj_c1 = (obj_c1 - self.best_f.unsqueeze(-1).to(obj_c1)).clamp_min(0)
        #     q_ei_c1 = obj_c1.max(dim=-1)[0].mean(dim=0)
        #     return q_ei_c1
        # elif self.fc == 2:
        #     obj_c2 = self.objective(samples[::2], X=X)
        #     obj_c2 = (obj_c2 - self.best_f.unsqueeze(-1).to(obj_c2)).clamp_min(0)
        #     q_ei_c2 = obj_c2.max(dim=-1)[0].mean(dim=0)
        #     return q_ei_c2
        obj = self.objective(samples, X=X)
        if self.fc == 0:
            obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
            q_ei = obj.max(dim=-1)[0].mean(dim=0)
        elif self.fc == 1:
            obj_c1 = obj[1::2]
            obj_c1 = (obj_c1 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
            q_ei = obj_c1.max(dim=-1)[0].mean(dim=0)
        elif self.fc == 2:
            obj_c2 = obj[::2]
            obj_c2 = (obj_c2 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
            q_ei = obj_c2.max(dim=-1)[0].mean(dim=0)
        return q_ei

