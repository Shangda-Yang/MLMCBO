import os
import numpy as np
import pandas as pd
import torch
from botorch.test_functions import Ackley, Hartmann
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from mlmcbo.utils.latin_hypercube_generator import generate_latin_hypercube_points
from mlmcbo.utils.model_fit import GPmodel
from mlmcbo.utils.objectiveFunctions import SelfDefinedFunction, Ackley5, SixHumpCamel2, Cosine, Levy10, Branin2
import warnings
from runBO import runBO, runBOTwo

import faulthandler

from tutorials.error_matrics import GAP, compute_nmse, compute_regret

faulthandler.enable()

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_default_dtype(torch.double)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(66)

# target = Hartmann(negate=True)
# fun_name = "Hartmann6"
target = Ackley(dim=2, negate=True)
fun_name = "Ackley2"


# ----------------------------------------
# new target
# target = Branin2(negate=True)
# fun_name = "BraninCompare"
# target = SixHumpCamel2(negate=True)
# fun_name = "SixHumpCamel2"
# target = Ackley5(negate=True)
# fun_name = "Ackley5"
# target = Cosine(negate=True)
# fun_name = "Cosine8"
# target = Levy10(negate=True)
# fun_name = "Levy10"

# boundary of the problem
bounds = target.bounds
lower_bounds, upper_bounds = bounds[0], bounds[1]

# dimension of the problem
dim = target.dim

# number of initial observations
num_obs = 2 * dim

# number of BO runs
n_runs = 15

# parameters for LBFGS
num_restarts = 20  # number of restarts
raw_samples = 256  # number of raw samples for each restarts

# number of realisations
R = 5

# starting level of MLMC
dl = 3

# initialise results storage
results_ml = torch.zeros(R, n_runs)
costs_ml = torch.zeros(R, n_runs)

results_sl = torch.zeros(R, n_runs)
costs_sl = torch.zeros(R, n_runs)

# reference solution
reference = target.optimal_value

# predefined accuracy - \varepsilon
eps = 0.2

# initialise relative benchmark for NMSE
relative = torch.zeros(R, 1)
# initialise relative benchmark for GAP
relative_gap = torch.zeros(R, 1)

# match mode for MLMC - 'point', 'forward', 'backward'
match_mode = 'point'
# kernel - 'Matern', 'RBF'
kernel = 'Matern'

for i in range(R):
    # generate initial observations
    train_x = (upper_bounds - lower_bounds) * torch.rand(num_obs, dim, device=device, dtype=torch.double) + lower_bounds
    # used for simple regret as Frazier's work
    # train_x = torch.tensor(generate_latin_hypercube_points(num_obs, bounds.reshape((dim, -1)).numpy()), device=device)
    train_y = target(train_x).unsqueeze(-1)

    # denominater of NMSE
    relative[i] = (torch.max(train_y) - reference) ** 2
    relative_gap[i] = torch.max(train_y)

    print("Realisation {}.".format(i))
    print("--------------------------------------------")
    print("MLMC starts")
    print("********************************************")

    # MLMC BO run
    # q is for number of batch size - qEI; for example, q = [2, 2] is qEI + qEI
    # ML=True - MLMC; ML=False - MC
    # match_mode = 'point', 'forward' or 'backward'
    bo_mlmc = runBOTwo(target=target,
                        train_x=train_x,
                        train_y=train_y,
                        n_runs=n_runs,
                        bounds=bounds,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        eps=eps,
                        q=[1, 1, 1],
                        ML=True,
                        dl=dl,
                        match_mode=match_mode,
                        kernel=kernel)
    results_ml[i, :], costs_ml[i, :] = bo_mlmc.run()
    #
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
                  q=[1, 2],
                  ML=False,
                  kernel=kernel)
    results_sl[i, :], costs_sl[i, :] = bo_mc.run()


# NMSE
MSE_ml, errorBar_ml, Cost_ml = compute_nmse(results_ml, costs_ml, reference, relative, R)
MSE_sl, errorBar_sl, Cost_sl = compute_nmse(results_sl, costs_sl, reference, relative, R)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)
ax.errorbar(Cost_ml, MSE_ml, xerr=None, yerr=errorBar_ml, fmt='--o', capsize=3)
ax.errorbar(Cost_sl, MSE_sl, xerr=None, yerr=errorBar_sl, fmt='--o', capsize=3)
ax.grid()
ax.legend(["MLMC2LAEI", "MC12LAEI"], fontsize=20, loc="lower left")
ax.set_xlabel("Expected cumulative wall time in second", fontsize=20)
ax.set_ylabel("NMSE", fontsize=20)
ax.tick_params(axis='both', labelsize=15)
ax.set_yscale("log")
ax.xaxis.get_major_formatter()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.show()

# GAP
mean_gap_ml, median_gap_ml = GAP(results_ml, relative_gap, reference, R)
mean_gap_sl, median_gap_sl = GAP(results_sl, relative_gap, reference, R)

print("{} ML2LA GAP Mean : {:.5f}, GAP Median : {:.5f}".format(fun_name, mean_gap_ml[-1], median_gap_ml[-1]))
print("{} SL2LA GAP Mean : {:.5f}, GAP Median : {:.5f}".format(fun_name, mean_gap_sl[-1], median_gap_sl[-1]))

# simple regret
regret_ml, errorBar_ml = compute_regret(results_ml, reference, fun_name, num_obs, n_runs, R, ML=True)
regret_sl, errorBar_sl = compute_regret(results_sl, reference, fun_name, num_obs, n_runs, R, ML=False)
func_eva = torch.arange(n_runs) + num_obs

ymax = np.floor(np.log10(max(max(regret_ml + errorBar_ml), max(regret_sl + errorBar_sl))))
ymin = np.ceil(np.log10(min(min(regret_ml - errorBar_ml), min(regret_sl - errorBar_sl))))

bar_ml = torch.log10(1 + errorBar_ml/regret_ml)
bar_sl = torch.log10(1 + errorBar_sl/regret_ml)

bar_ml[bar_ml < 0] = 0
bar_sl[bar_sl < 0] = 0

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)
ax.errorbar(func_eva, torch.log10(regret_ml), xerr=None, yerr=bar_ml, fmt='--o', capsize=3)
ax.errorbar(func_eva, torch.log10(regret_sl), xerr=None, yerr=bar_sl, fmt='--o', capsize=3)
ax.grid()
ax.legend(["MLMC2LAEI", "MC12LAEI"], fontsize=20, loc="upper right")
ax.set_xlabel("Function Evaluations", fontsize=20)
ax.set_ylabel("$\log$10(Simple Regret)", fontsize=20)
ax.tick_params(axis='both', labelsize=20)
# ax.set_ylim([ymin, ymax])
ax.xaxis.get_major_formatter()
# ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# plt.savefig("../Figures/Regret{}".format(fun_name))
plt.show()