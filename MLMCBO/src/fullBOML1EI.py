import pickle
import time
import numpy as np
import torch
from botorch.test_functions import Shekel
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mlmc_run import mlmc_run
from inc_fn import test_fn_ant_inc
from ObjectiveFunction import SelfDefinedFunction
import os

from mc_run import mc_run

# This script run full BO with one-step lookahead function
# TO DO: functional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.manual_seed(66)
dim = 1
num_obs = 2
target = SelfDefinedFunction()
fun_name = "SelfDefinedFunction"
bounds = target.bounds.to(dtype)
lower_bounds, upper_bounds = bounds[0], bounds[1]

train_X_sl = (upper_bounds-lower_bounds)*torch.rand(num_obs, dim, device=device, dtype=dtype) + lower_bounds
train_Y_sl = target(train_X_sl).unsqueeze(-1)

train_X_ml = train_X_sl
train_Y_ml = train_Y_sl

n_runs = 30

num_restarts = 20
raw_samples = 512

R = 10
dl = 3
alpha = 1
beta = 1.5
gamma = 1

kwargs_sl = {"train_x": train_X_sl,
            "train_y": train_Y_sl,
            "bounds": bounds,
            "num_restarts": num_restarts,
            "raw_samples": raw_samples
            }

kwargs_ml = {"train_x": train_X_ml,
            "train_y": train_Y_ml,
            "bounds": bounds,
            "num_restarts": num_restarts,
            "raw_samples": raw_samples
            }

# for i in range(3):
#     new_x, new_y, _ = mlmc_run(0.2, test_fn_ant_inc, dl, alpha, beta, gamma, target, **kwargs)
#     train_x = torch.cat([train_x, new_x])
#     train_y = torch.cat([train_y, new_y])
#     kwargs["train_x"] = train_x
#     kwargs["train_y"] = train_y

# kwargs = {"train_x": train_x,
#           "train_y": train_y,
#           "bounds": bounds,
#           "num_restarts": num_restarts,
#           "raw_samples": raw_samples
#           }

results_SL = torch.zeros(R, n_runs)
candidates_SL = torch.zeros(R, n_runs, dim)
costs_SL = torch.zeros(R, n_runs)

results_ML = torch.zeros(R, n_runs)
candidates_ML = torch.zeros(R, n_runs, dim)
costs_ML = torch.zeros(R, n_runs)

reference = target.optimal_value
eps_SL = 0.2
eps_ML = 0.2

# start_time = time.time()
for i in range(R):
    print("Realisation {}.".format(i))
    print("--------------------------------------------")
    kwargs_sl["train_x"] = train_X_sl
    kwargs_sl["train_y"] = train_Y_sl
    kwargs_ml["train_x"] = train_X_ml
    kwargs_ml["train_y"] = train_Y_ml
    train_x_sl = train_X_sl
    train_y_sl = train_Y_sl
    train_x_ml = train_X_ml
    train_y_ml = train_Y_ml
    for j in range(n_runs):
        start_time = time.time()
        # new_candidate_SL, new_result_SL, _ = mlmc_run(eps_SL, test_fn_ant_inc, dl, alpha, beta, gamma, target, **kwargs_ml)
        new_candidate_SL, new_result_SL, _ = mc_run(eps_SL, alpha, target, **kwargs_sl)
        cost_SL = time.time() - start_time

        start_time = time.time()
        new_candidate_ML, new_result_ML, _ = mlmc_run(eps_ML, test_fn_ant_inc, dl, alpha, beta, gamma, target, **kwargs_ml)
        cost_ML = time.time() - start_time

        train_x_sl = torch.cat([train_x_sl, new_candidate_SL])
        train_y_sl = torch.cat([train_y_sl, new_result_SL])

        kwargs_sl["train_x"] = train_x_sl
        kwargs_sl["train_y"] = train_y_sl

        train_x_ml = torch.cat([train_x_sl, new_candidate_ML])
        train_y_ml = torch.cat([train_y_sl, new_result_ML])

        kwargs_ml["train_x"] = train_x_ml
        kwargs_ml["train_y"] = train_y_ml

        best_candidate_sl = train_x_sl[torch.argmax(train_y_sl)]
        best_result_sl = target(best_candidate_sl).unsqueeze(-1)

        best_candidate_ml = train_x_ml[torch.argmax(train_y_ml)]
        best_result_ml = target(best_candidate_ml).unsqueeze(-1)

        results_SL[i, j] = best_result_sl
        results_ML[i, j] = best_result_ml

        costs_SL[i, j] = cost_SL
        costs_ML[i, j] = cost_ML
        print("Iteration {} finished, time {}".format(j, cost_SL + cost_ML))
# end_time = time.time()

error_SL = results_SL.cpu() - reference
error_ML = results_ML.cpu() - reference
MSE_SL = torch.sum(error_SL**2, dim=0) / R
MSE_ML = torch.sum(error_ML**2, dim=0) / R
errorBar_SL = 1.96 * torch.std(np.log(error_SL**2), dim=0)/np.sqrt(len(MSE_SL))
errorBar_ML = 1.96 * torch.std(np.log(error_ML**2), dim=0)/np.sqrt(len(MSE_ML))

Cost_SL = torch.mean(torch.cumsum(costs_SL, dim=1), dim=0)
Cost_ML = torch.mean(torch.cumsum(costs_ML, dim=1), dim=0)
costBar_SL = 1.96 * torch.std(torch.cumsum(costs_SL, dim=1))/np.sqrt(len(Cost_SL))
costBar_ML = 1.96 * torch.std(torch.cumsum(costs_ML, dim=1))/np.sqrt(len(Cost_ML))

results_fullBO = {"Results_SL": results_SL, "Results_ML": results_ML,
                "MSE_SL": MSE_SL, "MSE_ML": MSE_ML,
                "Cost_SL": Cost_SL, "Cost_ML": Cost_ML,
                "errorBar_SL": errorBar_SL, "errorBar_ML": errorBar_ML,
                "costBar_SL": costBar_SL, "costBar_ML": costBar_ML,
                "NoB": num_obs,
                "Runs": n_runs}

# results = {"MSE_ML": MSE_ML,
#            "Cost_ML": Cost_ML,
#            "Rate_ML": rate_ml}
with open('results_fullBO.pkl', 'wb') as file:
    pickle.dump(results_fullBO, file)

# plot
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
fig1.tight_layout(pad=10.0)
# ax.plot(Cost_SL, MSE_SL, '--.', linewidth=3, markersize=12)
# ax1.plot(range(num_obs + 1, num_obs + n_runs + 1), np.log(MSE_ML), '--*', linewidth=3, markersize=12)
# ax1.plot(range(num_obs + 1, num_obs + n_runs + 1), np.log(MSE_SL), '--o', linewidth=3, markersize=12)
ax1.errorbar(range(num_obs + 1, num_obs + n_runs + 1), np.log(MSE_ML), xerr=None, yerr=errorBar_ML, fmt='--o', capsize=3)
ax1.errorbar(range(num_obs + 1, num_obs + n_runs + 1), np.log(MSE_SL), xerr=None, yerr=errorBar_SL, fmt='--o', capsize=3)
ax1.grid()
ax1.legend(["MLMC", "MC"], fontsize=20)
ax1.set_xlabel("Number of observations", fontsize=20)
ax1.set_ylabel("MSE of suggested point in log scale", fontsize=20)
ax1.tick_params(axis='both', labelsize=15)
# ax1.tick_params(axis='x', rotation=45)
ax1.set_xlim([0, num_obs + n_runs + 1])
ax1.set_ylim([torch.min(torch.min(np.log(MSE_ML), np.log(MSE_SL))) - 1, 0])
# ax1.set_ylim([0, torch.ceil(torch.max(torch.max(MSE_SL, MSE_ML)))])
ax1.xaxis.get_major_formatter()
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
plt.savefig("1stepEIfullBOMSE{}R{}Init{}.eps".format(fun_name, R, num_obs), format='eps')
# plt.savefig("onestepMSE{}Iter{}R{}.eps".format(fun_name, n_runs, R), format='eps')
# plt.show()

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
fig2.tight_layout(pad=10.0)
# ax.plot(Cost_SL, MSE_SL, '--.', linewidth=3, markersize=12)
# ax2.plot(range(num_obs + 1, num_obs + n_runs + 1), Cost_ML, '--*', linewidth=3, markersize=12)
# ax2.plot(range(num_obs + 1, num_obs + n_runs + 1), Cost_SL, '--o', linewidth=3, markersize=12)
ax2.errorbar(range(num_obs + 1, num_obs + n_runs + 1), Cost_ML, xerr=None, yerr=costBar_ML, fmt='--o', capsize=3)
ax2.errorbar(range(num_obs + 1, num_obs + n_runs + 1), Cost_SL, xerr=None, yerr=costBar_SL, fmt='--o', capsize=3)
ax2.grid()
ax2.legend(["MLMC", "MC"], fontsize=20)
ax2.set_xlabel("Number of observations", fontsize=20)
ax2.set_ylabel("Cumulative wall time in second", fontsize=20)
ax2.tick_params(axis='both', labelsize=15)
# ax2.tick_params(axis='x', rotation=45)
ax2.set_xlim([0, num_obs + n_runs + 1])
ax2.set_ylim([0, torch.ceil(torch.max(torch.max(Cost_SL + torch.max(costBar_SL), Cost_ML + torch.max(costBar_ML)))) + 1])
ax2.xaxis.get_major_formatter()
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.savefig("1stepEIfullBOTime{}R{}Init{}.eps".format(fun_name, R, num_obs), format='eps')


fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
fig3.tight_layout(pad=10.0)
# ax.plot(Cost_SL, MSE_SL, '--.', linewidth=3, markersize=12)
# ax2.plot(range(num_obs + 1, num_obs + n_runs + 1), Cost_ML, '--*', linewidth=3, markersize=12)
# ax2.plot(range(num_obs + 1, num_obs + n_runs + 1), Cost_SL, '--o', linewidth=3, markersize=12)
ax3.errorbar(Cost_ML, np.log(MSE_ML),  xerr=None, yerr=errorBar_ML, fmt='--o', capsize=3)
ax3.errorbar(Cost_SL, np.log(MSE_SL),  xerr=None, yerr=errorBar_SL, fmt='--o', capsize=3)
# ax3.loglog(Cost_ML, MSE_ML, '--o')
# ax3.loglog(Cost_SL, MSE_SL, '--o')
ax3.grid()
ax3.legend(["MLMC", "MC"], fontsize=20)
ax3.set_xlabel("Cumulative wall time in second", fontsize=20)
ax3.set_ylabel("MSE of suggested point in log scale", fontsize=20)
ax3.tick_params(axis='both', labelsize=15)
# ax2.tick_params(axis='x', rotation=45)
# ax3.set_xlim([0, num_obs + n_runs + 1])
# ax3.set_ylim([0, torch.ceil(torch.max(torch.max(Cost_SL + torch.max(costBar_SL), Cost_ML + torch.max(costBar_ML)))) + 1])
ax3.xaxis.get_major_formatter()
# ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.savefig("1stepEIfullBOMSEvsCost{}Iter{}R{}.eps".format(fun_name, n_runs, R), format='eps')




