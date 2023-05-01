import torch
from typing import Optional, Tuple, Union, Any

from botorch import settings
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, qKnowledgeGradient, \
    MCAcquisitionObjective, qExpectedImprovement
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, IIDNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Tensor


class OneStepEI(qKnowledgeGradient):
    r"""one-step lookahead expected improvement
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

    @t_batch_mode_transform()
    def evaluate(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> Tensor:
        r"""Evaluate one-step lookahead EI on the candidate set `X_actual` by
            solving the inner optimization problem.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points
                each. Unlike `forward()`, this does not include solutions of the
                inner optimization problem.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of
                the solutions to the inner problem.
            kwargs: Additional keyword arguments. This includes the options for
                optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
                an `options` dictionary to be passed on to the optimization helpers, and
                a `scipy_options` dictionary to be passed to `scipy.minimize`.

        Returns:
            A Tensor of shape `b`. For t-batch b, the one-step lookahead EI value
                of the design `X[b]` is averaged across the fantasy models.
        """

        current_value = (
            self.current_value if self.current_value else self.model.train_targets.max()
        )

        ei = ExpectedImprovement(model=self.model, best_f=current_value)
        zero_step_ei = ei(torch.permute(X, (1, 0, -1)))

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X = torch.cat(
                [X, match_batch_shape(self.X_pending, X)], dim=-2
            )

        # construct the fantasy model of shape 'num_fantasies x b'
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=False
        )

        best_f = fantasy_model.train_targets.max(dim=-1)[0]

        if not self.inner_sampler:
            one_step_ei = ExpectedImprovement(model=fantasy_model, best_f=best_f)
        else:
            one_step_ei = qExpectedImprovement(model=fantasy_model,
                                               sampler=self.inner_sampler,
                                               best_f=best_f)

        from botorch.generation.gen import gen_candidates_scipy

        # optimize the inner problem
        from botorch.optim.initializers import gen_value_function_initial_conditions

        initial_conditions = gen_value_function_initial_conditions(
            acq_function=one_step_ei,
            bounds=bounds,
            num_restarts=kwargs.get("num_restarts", 20),
            raw_samples=kwargs.get("raw_samples", 1024),
            current_model=self.model,
            options={**kwargs.get("options", {}), **kwargs.get("scipy_options", {})},
        )

        _, values = gen_candidates_scipy(
            initial_conditions=initial_conditions,
            acquisition_function=one_step_ei,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options=kwargs.get("scipy_options"),
        )

        # get the maximizer for each batch
        values, _ = torch.max(values, dim=0)

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


class OneStepPI(qKnowledgeGradient):
    r"""one-step lookahead probability of improvement
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
        r"""Evaluate one step-lookahead PI on the candidate set 'X'

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

        pi = ProbabilityOfImprovement(model=self.model, best_f=current_value)
        zero_step_pi = pi(X_actual)

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

        one_step_pi = ProbabilityOfImprovement(model=fantasy_model, best_f=best_f)

        with settings.propagate_grads(True):
            values = one_step_pi(X=X_fantasies)

        one_step_pi_avg = values.mean(dim=0)

        return zero_step_pi + one_step_pi_avg

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


