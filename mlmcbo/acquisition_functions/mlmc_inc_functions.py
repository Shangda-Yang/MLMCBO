import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    MCAcquisitionObjective,
    qExpectedImprovement,
    qMultiStepLookahead, PosteriorMean, MCAcquisitionFunction,
)
from botorch.acquisition.multi_step_lookahead import make_best_f, _construct_sample_weights, _construct_inner_samplers, \
    _compute_stage_value
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


class qEIMLMCTwoStep:
    r"""Class for two-step lookahead q-EI with MLMC (2-EI) by default"""

    def __init__(self, model, bounds, num_restarts, raw_samples, q=1, batch_sizes=None):
        r"""
        Args:
            bounds: bounds of objective function
            model: a fitted single-outcome model
            num_restarts: number of restarts for LBFGS
            raw_samples: number of raw samples for LBFGS
            q: numer of batch observations generated each iteration
            batch_sizes: array for q of q-EI, default is [2] (2-EI)
        """
        if batch_sizes is None:
            batch_sizes = [2]
        self.model = model
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.q = q
        self.batch_sizes = batch_sizes

    @staticmethod
    def _create_sampler(num_samples, seed=None):
        r"""Initialize sampler with a seed for coupling"""
        return IIDNormalSampler(sample_shape=torch.Size([num_samples]), seed=seed)

    def sample_candidate(self, l, dl, num_samples, match=None, match_mode='point'):
        r"""Generate new observations with MLMC
        Args:
            l: index level of estimation (starting from 0)
            dl: starting level (l <- l + dl)
            num_samples: number of outer samples
            match: if None, set the initial matched value
                   else matching the optimizer, so that MLMC tracks the same optimizer among levels;
            match_mode: 'point' or 'forward' or 'backward'
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
        # seed_out, seed_in = (torch.randint(0, 10000000, (1,)).item() for _ in range(2))

        sampler = self._create_sampler(num_samples)

        samplers = [sampler]
        inner_mc_samplers = [None if self.q == 1 else sampler, self._create_sampler(M)]

        if l == 0:
            if match_mode == 'point':
                new_candidate, new_value = self.get_candidates(samplers, inner_mc_samplers, return_best=True)
                match_candidate = new_candidate
            elif match_mode == 'forward':
                new_candidate, new_value = self.get_candidates(samplers, inner_mc_samplers, return_best=False)
                match_candidate = new_candidate
            elif match_mode == 'backward':
                new_candidate, new_value = self.get_candidates(samplers, inner_mc_samplers, return_best=False)
                diff = torch.argmin(dist_matrix(match, new_candidate), dim=1)
                new_candidate = new_candidate[diff]
                new_value = new_value[diff]
                match_candidate = new_candidate[diff]
        else:
            new_candidate_f, new_value_f = self.get_candidates(samplers, inner_mc_samplers,
                                                               return_best=False)
            antithetic = True

            if not antithetic:
                inner_sampler = self._create_sampler(M//2)
                inner_sampler.base_samples = inner_mc_samplers[1].base_samples[::2]
                inner_mc_samplers[1] = inner_sampler
                
            new_candidate_c, new_value_c = self.get_candidates(samplers, inner_mc_samplers,
                                                               antithetic=antithetic,
                                                               return_best=False)
            if match_mode == 'point':
                diff_c = torch.argmin(torch.norm(match - new_candidate_c, dim=[1, 2]))
                diff_f = torch.argmin(torch.norm(new_candidate_f - new_candidate_c[diff_c], dim=[1, 2]))
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_f[diff_f]
            elif match_mode == 'forward':
                diff_c = torch.argmin(dist_matrix(match, new_candidate_c), dim=1)
                diff_f = torch.argmin(dist_matrix(new_candidate_c[diff_c], new_candidate_f), dim=1)
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_f[diff_f]
            elif match_mode == 'backward':
                if match is None:
                    match = new_candidate_f
                diff_f = torch.argmin(dist_matrix(match, new_candidate_f), dim=1)
                diff_c = torch.argmin(dist_matrix(new_candidate_f[diff_f], new_candidate_c), dim=1)
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_c[diff_c]

        return new_candidate, new_value, match_candidate

    def get_candidates(self, samplers, inner_mc_samplers, antithetic=False, return_best=True):
        r"""Generate the next observation of single two-step lookahead EI.
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
        valfunc_cls = [ExpectedImprovement if self.q == 1 else qExpectedImprovement,
                       qExpectedImprovement if antithetic is False else qExpectedImprovementAnt]
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

        q_prime = oneqEI.get_augmented_q_batch_size(self.q)
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
        # obj = self.objective(samples, X=X)

        # # separate the samples and compute each estimation for antithetic approach
        # obj_1 = obj[1::2]
        # obj_1 = (obj_1 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        # q_ei_1 = obj_1.max(dim=-1)[0].mean(dim=0)
        # obj_2 = obj[::2]
        # obj_2 = (obj_2 - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        # q_ei_2 = obj_2.max(dim=-1)[0].mean(dim=0)
        # return (q_ei_1 + q_ei_2) / 2

        obj_1 = self.objective(samples[::2], X=X)
        obj_1 = (obj_1 - self.best_f.unsqueeze(-1).to(obj_1)).clamp_min(0)
        q_ei_1 = obj_1.max(dim=-1)[0].mean(dim=0)

        obj_2 = self.objective(samples[1::2], X=X)
        obj_2 = (obj_2 - self.best_f.unsqueeze(-1).to(obj_2)).clamp_min(0)
        q_ei_2 = obj_2.max(dim=-1)[0].mean(dim=0)

        inds_1 = torch.argmax(q_ei_1, dim=-1)
        inds_2 = torch.argmax(q_ei_2, dim=-1)

        for i in range(len(inds_1)):
            temp = q_ei_2[i, inds_1[i]].item()
            q_ei_2[i, inds_1[i]] = q_ei_2[i, inds_2[i]]
            q_ei_2[i, inds_2[i]] = temp

        return (q_ei_1 + q_ei_2)/2

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of improvement utility values.
        """
        return (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)


class qEIMLMCThreeStep:
    r"""Class for three-step lookahead q-EI with MLMC (1-EI) by default"""

    def __init__(self, model, bounds, num_restarts, raw_samples, q=1, batch_sizes=None):
        r"""
        Args:
            bounds: bounds of objective function
            model: a fitted single-outcome model
            num_restarts: number of restarts for LBFGS
            raw_samples: number of raw samples for LBFGS
            q: numer of batch observations generated each iteration
            batch_sizes: array for q of q-EI, default is [1, 1] (1-EI)
        """
        if batch_sizes is None:
            batch_sizes = [1, 1]
        self.model = model
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.q = q
        self.batch_sizes = batch_sizes

    @staticmethod
    def _create_sampler(num_samples, seed=None):
        r"""Initialize sampler with a seed for coupling"""
        return IIDNormalSampler(sample_shape=torch.Size([num_samples]), seed=seed)

    def sample_candidate(self, l, dl, num_samples, match=None, match_mode='point'):
        r"""Generate new observations with MLMC
        Args:
            l: index level of estimation (starting from 0)
            dl: starting level (l <- l + dl)
            num_samples: number of outer samples
            match: if None, set the initial matched value
                   else matching the optimizer, so that MLMC tracks the same optimizer among levels;
            match_mode: 'point' or 'forward' or 'backward'
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
        # seed_out, seed_in = (torch.randint(0, 10000000, (1,)).item() for _ in range(2))

        samplers = [self._create_sampler(num_samples), self._create_sampler(M)]

        if l == 0:
            if match_mode == 'point':
                new_candidate, new_value = self.get_candidates(samplers, return_best=True)
                match_candidate = new_candidate
            elif match_mode == 'forward':
                new_candidate, new_value = self.get_candidates(samplers, return_best=False)
                match_candidate = new_candidate
            elif match_mode == 'backward':
                new_candidate, new_value = self.get_candidates(samplers, return_best=False)
                diff = torch.argmin(dist_matrix(match, new_candidate), dim=1)
                new_candidate = new_candidate[diff]
                new_value = new_value[diff]
                match_candidate = new_candidate[diff]
        else:
            new_candidate_f, new_value_f = self.get_candidates(samplers,
                                                               return_best=False)
            new_candidate_c, new_value_c = self.get_candidates(samplers,
                                                               antithetic=True,
                                                               return_best=False)
            if match_mode == 'point':
                diff_c = torch.argmin(torch.norm(match - new_candidate_c, dim=[1, 2]))
                diff_f = torch.argmin(torch.norm(new_candidate_f - new_candidate_c[diff_c], dim=[1, 2]))
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_f[diff_f]
            elif match_mode == 'forward':
                diff_c = torch.argmin(dist_matrix(match, new_candidate_c), dim=1)
                diff_f = torch.argmin(dist_matrix(new_candidate_c[diff_c], new_candidate_f), dim=1)
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_f[diff_f]
            elif match_mode == 'backward':
                if match is None:
                    match = new_candidate_f
                diff_f = torch.argmin(dist_matrix(match, new_candidate_f), dim=1)
                diff_c = torch.argmin(dist_matrix(new_candidate_f[diff_f], new_candidate_c), dim=1)
                new_candidate = new_candidate_f[diff_f] - new_candidate_c[diff_c]
                new_value = new_value_f[diff_f] - new_value_c[diff_c]
                match_candidate = new_candidate_c[diff_c]

        return new_candidate, new_value, match_candidate

    def get_candidates(self, samplers, antithetic=False, return_best=True):
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

        # initialize the acquisition function for the zero, first and second steps
        valfunc_cls = [ExpectedImprovement,
                       ExpectedImprovement,
                       ExpectedImprovement]
        valfunc_argfacs = [make_best_f, make_best_f, make_best_f]

        if antithetic is False:
            twoqEI = CustomqMultiStepLookahead(
                model=self.model,
                batch_sizes=self.batch_sizes,
                num_fantasies=None,
                samplers=samplers,
                valfunc_cls=valfunc_cls,
                valfunc_argfacs=valfunc_argfacs,
            )
        else:
            twoqEI =ThreeStepIncAntEI(
                model=self.model,
                batch_sizes=self.batch_sizes,
                num_fantasies=None,
                samplers=samplers,
                valfunc_cls=valfunc_cls,
                valfunc_argfacs=valfunc_argfacs,
            )
        q_prime = twoqEI.get_augmented_q_batch_size(self.q)
        new_candidate, new_value = optimize_acqf(
            acq_function=twoqEI,
            bounds=self.bounds,
            q=q_prime,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            return_best_only=return_best
        )

        return new_candidate, new_value


class ThreeStepIncAntEI(qMultiStepLookahead):

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


    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiStepLookahead on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        Xs = self.get_multi_step_tree_input_representation(X)

        # set batch_range on samplers if not collapsing on fantasy dims
        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        return _antstep(
            model=self.model,
            Xs=Xs,
            samplers=self.samplers,
            valfunc_cls=self._valfunc_cls,
            valfunc_argfacs=self._valfunc_argfacs,
            inner_samplers=self.inner_samplers,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
        )

def _antstep(
    model: Model,
    Xs: List[Tensor],
    samplers: List[Optional[MCSampler]],
    valfunc_cls: List[Optional[Type[AcquisitionFunction]]],
    valfunc_argfacs: List[Optional[TAcqfArgConstructor]],
    inner_samplers: List[Optional[MCSampler]],
    objective: MCAcquisitionObjective,
    posterior_transform: PosteriorTransform,
) -> Tensor:
    r"""Antithetic estimator computation.
    """
    # compute zero step
    sample_weights = torch.ones(*Xs[0].shape[:-2], device=Xs[0].device, dtype=Xs[0].dtype)
    running_val = _compute_stage_value(
        model=model,
        valfunc_cls=valfunc_cls[0],
        X=Xs[0],
        objective=objective,
        posterior_transform=posterior_transform,
        inner_sampler=inner_samplers[0],
        arg_fac=valfunc_argfacs[0],
    )

    # construct fantasy points
    fantasy_model = model.fantasize(
        X=Xs[0], sampler=samplers[0], observation_noise=True, propagate_grads=False
    )

    # augment sample weights appropriately
    sample_weights = _construct_sample_weights(
        prev_weights=sample_weights, sampler=samplers[0]
    )

    # compute first step
    stage_val = _compute_stage_value(
        model=fantasy_model,
        valfunc_cls=valfunc_cls[1],
        X=Xs[1],
        objective=objective,
        posterior_transform=posterior_transform,
        inner_sampler=inner_samplers[1],
        arg_fac=valfunc_argfacs[1],
    )
    running_val = running_val + stage_val

    # construct fantasy points
    base_samples = samplers[1].base_samples
    base_samples1 = base_samples[0::2]
    base_samples2 = base_samples[1::2]
    sampler1 = IIDNormalSampler(sample_shape=torch.Size([len(base_samples1)]))
    sampler1.base_samples = base_samples1
    sampler2 = IIDNormalSampler(sample_shape=torch.Size([len(base_samples2)]))
    sampler2.base_samples = base_samples2

    fantasy_model1 = model.fantasize(
        X=Xs[1], sampler=sampler1, observation_noise=True, propagate_grads=True
    )
    fantasy_model2 = model.fantasize(
        X=Xs[1], sampler=sampler2, observation_noise=True, propagate_grads=True
    )

    # augment sample weights appropriately
    sample_weights = _construct_sample_weights(
        prev_weights=sample_weights, sampler=samplers[1]
    )

    # compute second step
    stage_val1 = _compute_stage_value(
        model=fantasy_model1,
        valfunc_cls=valfunc_cls[2],
        X=Xs[2][0::2],
        objective=objective,
        posterior_transform=posterior_transform,
        inner_sampler=inner_samplers[2],
        arg_fac=valfunc_argfacs[2],
    )

    stage_val2 = _compute_stage_value(
        model=fantasy_model2,
        valfunc_cls=valfunc_cls[2],
        X=Xs[2][1::2],
        objective=objective,
        posterior_transform=posterior_transform,
        inner_sampler=inner_samplers[2],
        arg_fac=valfunc_argfacs[2],
    )

    running_val = running_val + 1/2 * torch.cat((stage_val1, stage_val2), dim=0)

    batch_shape = running_val.shape[2:]
    # expand sample weights to make sure it is the same shape as running_val,
    # because we need to take a sum over sample weights for computing the
    # weighted average
    sample_weights = sample_weights.expand(running_val.shape)
    return (running_val * sample_weights).view(-1, *batch_shape).sum(dim=0)



def dist_matrix(x, y):
    # compute the distance matrix of two vectors
    xsq = torch.sum(x ** 2, dim=(1, 2))  # (N,)
    ysq = torch.sum(y ** 2, dim=(1, 2))  # (M,)

    mixprod = -2 * x.view(x.shape[0], -1) @ y.view(y.shape[0], -1).T  # (N, M)

    return torch.sqrt(xsq.unsqueeze(1) + mixprod + ysq.unsqueeze(0))  # (N,1)+(N,M)+(1,M) => (N, M)