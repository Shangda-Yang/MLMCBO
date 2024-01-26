import os
import numpy as np
import torch
from botorch.test_functions import Ackley, Hartmann
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from mlmcbo.utils.model_fit import GPmodel
from mlmcbo.utils.objectiveFunctions import SelfDefinedFunction
import warnings
from runBO import runBO

import faulthandler
faulthandler.enable()

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

results_ml = torch.zeros(R, n_runs)
costs_ml = torch.zeros(R, n_runs)

results_sl = torch.zeros(R, n_runs)
costs_sl = torch.zeros(R, n_runs)

reference = target.optimal_value
eps = 0.2

relative = torch.zeros(R, 1)

for i in range(R):
    train_x = (upper_bounds - lower_bounds) * torch.rand(num_obs, dim, device=device, dtype=torch.double) + lower_bounds
    train_y = target(train_x).unsqueeze(-1)

    relative[i] = (torch.max(train_y) - reference) ** 2

    print("Realisation {}.".format(i))
    print("--------------------------------------------")
    print("MLMC starts")
    print("********************************************")

    bo_mlmc = runBO(target=target,
                    train_x=train_x,
                    train_y=train_y,
                    n_runs=n_runs,
                    bounds=bounds,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    eps=eps,
                    q=1,
                    ML=True,
                    dl=dl)
    results_ml[i, :], costs_ml[i, :] = bo_mlmc.run()

    print("MC starts")
    print("********************************************")
    bo_mc = runBO(target=target,
                  train_x=train_x,
                  train_y=train_y,
                  n_runs=n_runs,
                  bounds=bounds,
                  num_restarts=num_restarts,
                  raw_samples=raw_samples,
                  eps=eps,
                  q=1,
                  ML=False)
    results_sl[i, :], costs_sl[i, :] = bo_mc.run()


def compute_metric(results, costs, reference, relative):
    # compute the NMSE, error bar, cumulative costs
    error = (results.cpu() - reference)
    MSE = torch.sum(error ** 2 / relative, dim=0) / R
    errorBar = 1.96 * torch.std(error ** 2 / relative, dim=0) / np.sqrt(len(MSE))
    Cost = torch.mean(torch.cumsum(costs, dim=1), dim=0)
    return MSE, errorBar, Cost

MSE_ml, errorBar_ml, Cost_ml = compute_metric(results_ml, costs_ml, reference, relative)
MSE_sl, errorBar_sl, Cost_sl = compute_metric(results_sl, costs_sl, reference, relative)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)
ax.errorbar(Cost_ml, MSE_ml, xerr=None, yerr=errorBar_ml, fmt='--o', capsize=3)
ax.errorbar(Cost_sl, MSE_sl, xerr=None, yerr=errorBar_sl, fmt='--o', capsize=3)
ax.grid()
ax.legend(["MLMC", "MC"], fontsize=20, loc="lower left")
ax.set_xlabel("Expected cumulative wall time in second", fontsize=20)
ax.set_ylabel("NMSE", fontsize=20)
ax.tick_params(axis='both', labelsize=15)
ax.set_yscale("log")
ax.xaxis.get_major_formatter()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.show()
