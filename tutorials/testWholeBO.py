import os
import time
from typing import Any, Callable, Dict, Optional, List, Tuple

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.models.model import Model
from botorch.test_functions import Hartmann, SyntheticTestFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from mlmcbo.acquisition_functions.mlmc_inc_functions import qEIMLMCOneStep, qEIMLMCTwoStep
from mlmcbo.utils.optimize_mlmc import optimize_mlmc

torch.set_default_dtype(torch.double)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(66)

class SelfDefinedFunction(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-10.0, 10.0)]
        self._optimal_value = 1.4019
        self._optimizers = [(2.0087)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.exp(-(X - 2) ** 2) + torch.exp(-(X - 6) ** 2 / 10) + 1 / (X ** 2 + 1)
        return f.squeeze(-1)
    # return -torch.sin(3*x) - x**2 + 0.7*x

target = SelfDefinedFunction()
fun_name = "SelfDefinedFunction"
# target = Hartmann(negate=True)
# fun_name = "Hartmann6"
bounds = target.bounds
lower_bounds, upper_bounds = bounds[0], bounds[1]

dim = target.dim
num_obs = 2*dim

train_X = (upper_bounds-lower_bounds)*torch.rand(num_obs, dim, device=device) + lower_bounds
train_Y = target(train_X).unsqueeze(-1)

n_runs = 20

num_restarts = 20
raw_samples = 512

R = 20
dl = 3
alpha = 1
beta = 1.5
gamma = 1

results = torch.zeros(R, n_runs)
candidates = torch.zeros(R, n_runs, dim)
costs = torch.zeros(R, n_runs)

reference = target.optimal_value
eps = 0.2

relative = torch.zeros(R, 1)
# start_time = time.time()
for i in range(R):
    train_x = (upper_bounds - lower_bounds) * torch.rand(num_obs, dim, device=device) + lower_bounds
    train_y = target(train_x).unsqueeze(-1)

    relative[i] = (torch.max(train_y) - reference)**2

    print("Realisation {}.".format(i))
    print("--------------------------------------------")

    for j in range(n_runs):
        train_yvar = torch.full_like(train_y, 1e-4)
        model = FixedNoiseGP(train_x, train_y, train_yvar)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        start_time = time.time()
        qEI = qEIMLMCOneStep(
            model=model,
            bounds=bounds,
            num_restarts=20,
            raw_samples=512
        )

        new_candidate, _, _ = optimize_mlmc(
            inc_function=qEI,
            eps=eps,
            dl=3,
            alpha=1,
            beta=1.5,
            gamma=1,
            meanc=1,
            varc=1,
            var0=1
        )
        new_result = target(new_candidate).unsqueeze(-1)

        cost = time.time() - start_time

        train_x = torch.cat([train_x, new_candidate])
        train_y = torch.cat([train_y, new_result])

        best_candidate = train_x[torch.argmax(train_y)]
        best_result = target(best_candidate).unsqueeze(-1)

        results[i, j] = best_result

        costs[i, j] = cost

        print("Iteration {} finished, MLMC time {:.4f}".format(j, cost))
        # print("Iteration {} finished, MLMC time {:.4f}".format(j, cost_ML))

# end_time = time.time()

error = (results.cpu() - reference)
MSE = torch.sum(error**2/relative, dim=0) / R
errorBar = 1.96 * torch.std(error**2/relative, dim=0)/np.sqrt(len(MSE))

Cost = torch.mean(torch.cumsum(costs, dim=1), dim=0)
costBar = 1.96 * torch.std(torch.cumsum(costs, dim=1))/np.sqrt(len(Cost))

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)
ax.errorbar(Cost, MSE,  xerr=None, yerr=errorBar, fmt='--o', capsize=3)
ax.grid()
ax.legend(["MLMC", "MC"], fontsize=20, loc="lower left")
ax.set_xlabel("Cumulative wall time in second", fontsize=20)
ax.set_ylabel("NMSE", fontsize=20)
ax.tick_params(axis='both', labelsize=15)
ax.set_yscale("log")
ax.xaxis.get_major_formatter()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.show()
