import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mlmcbo.acquisition_functions.mlmc_inc_functions import qEIMLMCOneStep
from mlmcbo.utils.objectiveFunctions import SelfDefinedFunction
from mlmcbo.utils.model_fit import GPmodel
from mlmcbo.utils.optimize_mlmc import optimize_mlmc

import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_default_dtype(torch.double)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(66)

target = SelfDefinedFunction()
fun_name = "SelfDefinedFunction"
# target = Hartmann(negate=True)
# fun_name = "Hartmann6"
# target = Ackley(dim=2, negate=True)
# fun_name = "Ackley2"

bounds = target.bounds
lower_bounds, upper_bounds = bounds[0], bounds[1]

dim = target.dim
num_obs = 2 * dim

n_runs = 30

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

for i in range(R):
    train_x = (upper_bounds - lower_bounds) * torch.rand(num_obs, dim, device=device, dtype=torch.double) + lower_bounds
    train_y = target(train_x).unsqueeze(-1)

    relative[i] = (torch.max(train_y) - reference) ** 2

    model = GPmodel(train_x, train_y)

    qEI = qEIMLMCOneStep(
        model=model,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples
    )

    print("Realisation {}.".format(i))
    print("--------------------------------------------")

    for j in range(n_runs):
        start_time = time.time()
        new_candidate, _, _ = optimize_mlmc(
            inc_function=qEI,
            eps=eps,
            dl=dl,
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

        model = GPmodel(train_x, train_y)
        qEI.model = model

        best_candidate = train_x[torch.argmax(train_y)]
        best_result = target(best_candidate).unsqueeze(-1)

        results[i, j] = best_result

        costs[i, j] = cost

        print("Iteration {} finished, MLMC time {:.4f}".format(j, cost))

error = (results.cpu() - reference)
MSE = torch.sum(error ** 2 / relative, dim=0) / R
errorBar = 1.96 * torch.std(error ** 2 / relative, dim=0) / np.sqrt(len(MSE))

Cost = torch.mean(torch.cumsum(costs, dim=1), dim=0)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)
ax.errorbar(Cost, MSE, xerr=None, yerr=errorBar, fmt='--o', capsize=3)
ax.grid()
ax.legend(["MLMC"], fontsize=20, loc="lower left")
ax.set_xlabel("Cumulative wall time in second", fontsize=20)
ax.set_ylabel("NMSE", fontsize=20)
ax.tick_params(axis='both', labelsize=15)
ax.set_yscale("log")
ax.xaxis.get_major_formatter()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.show()
