from typing import Any, Callable, Dict

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from mlmcbo.acquisition_functions.mc_one_step_lookahead import (qExpectedImprovementOneStepLookahead,
                                                                ExpectedImprovementOneStepLookahead)
from mlmcbo.acquisition_functions.mlmc_inc_functions import qEIMLMCOneStep, qEIMLMCTwoStep
from mlmcbo.utils.optimize_mlmc import optimize_mlmc, optimize_mlmc_two

TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]
torch.set_default_dtype(torch.double)

neg_hartmann6 = Hartmann(dim=6, negate=True)
torch.random.manual_seed(42)
bounds = neg_hartmann6.bounds
lower_bounds, upper_bounds = bounds[0], bounds[1]
train_x = (upper_bounds - lower_bounds) * torch.rand(10, 6) + lower_bounds
train_x = torch.rand(20, neg_hartmann6.dim)
train_obj = neg_hartmann6(train_x).unsqueeze(-1)

best_value = train_obj.max()
print(f'Hartmann example:\nOptimal value = {neg_hartmann6.optimal_value}\nOptimizers = {neg_hartmann6.optimizers}')

model = SingleTaskGP(
    train_X=train_x,
    train_Y=train_obj,
    input_transform=Normalize(d=neg_hartmann6.dim, bounds=bounds),
    outcome_transform=Standardize(m=1)
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# MC One-Step Lookahead EI
# ------------------------------------------------------------------------------------------
eps = 0.25
N = np.round((1 / np.power(eps, 2))).astype(int)
M = N

sampler = IIDNormalSampler(sample_shape=torch.Size([N]), resample=False)
inner_sampler = IIDNormalSampler(sample_shape=torch.Size([M]), resample=False)
EI = ExpectedImprovementOneStepLookahead(
    model=model,
    num_fantasies=None,
    sampler=sampler,
    inner_sampler=inner_sampler
)

torch.manual_seed(seed=0)
new_candidate, _ = optimize_acqf(
    acq_function=EI,
    bounds=bounds,
    q=1,
    num_restarts=30,
    raw_samples=100,
    options={},
)
print(f'New candidate [MC One-Step Lookahead EI] = {new_candidate}')

# MC One-Step Lookahead qEI
# ------------------------------------------------------------------------------------------
batch_sizes = [2]

qEI = qExpectedImprovementOneStepLookahead(
    model=model,
    batch_sizes=batch_sizes,
    antithetic_variates=False
)
n_points = qEI.get_augmented_q_batch_size(1)

torch.manual_seed(seed=0)
new_candidate, _ = optimize_acqf(
    acq_function=qEI,
    bounds=bounds,
    q=n_points,
    num_restarts=30,
    raw_samples=100,
    return_best_only=True,
    options={},
)
print(f'New candidate [MC One-Step Lookahead qEI] = {new_candidate}')

# MLMC One-Step Lookahead qEI
# ------------------------------------------------------------------------------------------
qEI = qEIMLMCOneStep(
    model=model,
    bounds=bounds,
    num_restarts=30,
    raw_samples=100,
    q=1,
    batch_sizes=[2]
)

torch.manual_seed(seed=0)
new_candidate, _, _ = optimize_mlmc(
    inc_function=qEI,
    eps=1e-1,
    dl=3,
    alpha=1,
    beta=1.5,
    gamma=1,
    meanc=1,
    varc=1,
    var0=1
)
print(f'New candidate [MLMC One-Step Lookahead qEI] = {new_candidate}')

# MLMC Two-Step Lookahead qEI
# ------------------------------------------------------------------------------------------
twoqEI = qEIMLMCTwoStep(
    model=model,
    bounds=bounds,
    num_restarts=10,
    raw_samples=256,
    q=1,
    batch_sizes=[1, 1]
)

new_candidate, _, _ = optimize_mlmc_two(
    inc_function=twoqEI,
    eps=0.1,
    dl=3,
    alpha=1,
    beta=1.5,
    gamma=1,
    meanc=1,
    varc=1,
    var0=1,
    match_mode='point'
)
print(f'New candidate [MLMC Two-Step Lookahead qEI] = {new_candidate}')

