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


# ------------------------------------------------------------- #
# test of mlmc complexity with 1-step EI
# TO DO: functional
# ------------------------------------------------------------- #

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.manual_seed(99)
dim = 1
num_obs = 6
target = SelfDefinedFunction()
fun_name = "SelfDefinedFunction"
bounds = target.bounds.to(dtype)
lower_bounds, upper_bounds = bounds[0], bounds[1]
train_x = (upper_bounds-lower_bounds)*torch.rand(num_obs, dim, device=device, dtype=dtype) + lower_bounds
train_y = target(train_x).unsqueeze(-1)

n_runs = 1

num_restarts = 20
raw_samples = 512

Eps_sl = [0.2, 0.22, 0.25, 0.3, 0.35, 0.42]
# Eps_sl = [5, 10, 15, 20, 25, 30, 35]
Eps_ml = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4]
R = 200
dl = 4
alpha = 1
beta = 1.5
gamma = 1

kwargs = {"train_x": train_x,
          "train_y": train_y,
          "bounds": bounds,
          "num_restarts": num_restarts,
          "raw_samples": raw_samples
          }

# for i in range(3):
#     new_x, new_y, _ = mlmc_run(0.1, test_fn_ant_inc, dl, alpha, beta, gamma, target, **kwargs)
#     train_x = torch.cat([train_x, new_x])
#     train_y = torch.cat([train_y, new_y])
#     kwargs["train_x"] = train_x
#     kwargs["train_y"] = train_y
#
# kwargs = {"train_x": train_x,
#           "train_y": train_y,
#           "bounds": bounds,
#           "num_restarts": num_restarts,
#           "raw_samples": raw_samples
#           }

results_SL = torch.zeros(R, len(Eps_sl))
candidates_SL = torch.zeros(R, len(Eps_sl), dim)
costs_SL = torch.zeros(R, len(Eps_sl))

results_ML = torch.zeros(R, len(Eps_ml))
candidates_ML = torch.zeros(R, len(Eps_ml), dim)
costs_ML = torch.zeros(R, len(Eps_ml))

# reference
# ref_candidate = 0.0
# ref_result = 0.0
# r = 10
# for j in range(r):
#     best_candidate, best_result, _ = mc_run(100, alpha, target, ref=True, **kwargs)
#     ref_result += best_result/r
#     ref_candidate += best_candidate/r
#     print("Reference {}".format(j))
# print("Reference finished.")
#
# with open('reference6obs.pkl', 'wb') as file:
#     pickle.dump(ref_candidate, file)
#
with open('reference6obs.pkl', 'rb') as file:
    reference = pickle.load(file)

for i in range(len(Eps_ml)):
    eps_sl = Eps_sl[i]
    eps_ml = Eps_ml[i]
    start_time = time.time()
    print("-------------------------")
    print("Eps {}".format(eps_ml))
    print("-------------------------")
    for j in range(R):
        for k in range(n_runs):
            # start = time.time()
            # new_candidate_SL, new_result_SL, C_SL = mc_run(eps_sl, alpha, target, ref=False, **kwargs)
            # cost = time.time() - start
            new_candidate, new_result, C_ML = mlmc_run(eps_ml, test_fn_ant_inc, dl, alpha, beta, gamma, target, **kwargs)
            # train_x = torch.cat([train_x, new_candidate])
            # train_y = torch.cat([train_y, new_result])
            #
            # kwargs["train_x"] = train_x
            # kwargs["train_y"] = train_y
            #
            # best_candidate = train_x[torch.argmax(train_y)]
            # best_result = target(best_candidate).unsqueeze(-1)

        # results_SL[j, i] = new_result_SL
        # candidates_SL[j, i, :] = new_candidate_SL
        # costs_SL[j, i] = C_SL

        results_ML[j, i] = new_result
        candidates_ML[j, i, :] = new_candidate
        costs_ML[j, i] = np.sum(C_ML)
        print("Realisation {} finished, time {}".format(j, time.time() - start_time))
end_time = time.time()
print("--- Time {:.4f} seconds ---".format(end_time - start_time))
reference = torch.mean(candidates_ML[:, 0])
# reference = ref_candidate
# MSE_SL = torch.sum((candidates_SL.cpu() - reference.cpu())**2, dim=0) / R
MSE_ML = torch.sum((candidates_ML.cpu() - reference.cpu())**2, dim=0) / R

# Cost_SL = torch.mean(costs_SL, dim=0)
Cost_ML = torch.mean(costs_ML, dim=0)

# rate_sl = np.polyfit(np.log(Cost_SL), np.log(MSE_SL.squeeze()), 1)[0]
# print("MC rate {:.4f}".format(rate_sl))

# results_slml = {"MSE_SL": MSE_SL, "MSE_ML": MSE_ML,
#            "Cost_SL": Cost_SL, "Cost_ML": Cost_ML,
#            "Rate_SL": rate_sl, "Rate_ML": rate_ml}

# results_sl = {"MSE": MSE_SL, "Cost": Cost_SL, "Rate": rate_sl, "Candidate": candidates_SL}

# results_ml = {"MSE": MSE_ML, "Cost": Cost_ML, "Rate": rate_ml, "Candidate": candidates_ML}
#
# # results = {"MSE_ML": MSE_ML,
# #            "Cost_ML": Cost_ML,
# #            "Rate_ML": rate_ml}

# with open('results6ob_ml.pkl', 'wb') as file:
#     pickle.dump(results_ml, file)

# with open('results6ob_sl.pkl', 'wb') as file:
#     pickle.dump(results_sl, file)
#
with open('results6ob_ml.pkl', 'rb') as file:
    results_ML = pickle.load(file)
candidates = results_ML['Candidate']
Cost_ML = results_ML['Cost']
MSE_ML = results_ML['MSE']

with open('results6ob_sl.pkl', 'rb') as file:
    results_SL = pickle.load(file)


Cost_SL = results_SL['Cost']
MSE_SL = results_SL['MSE']

rate_ml = np.polyfit(np.log(Cost_ML), np.log(MSE_ML.squeeze()), 1)[0]
print("MLMC rate {:.4f}".format(rate_ml))
# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)

# ax.plot(Cost_SL, MSE_SL, '--.', linewidth=3, markersize=12)
ax.loglog(Cost_ML, MSE_ML, '--*', linewidth=3, markersize=12)
ax.loglog(Cost_SL, MSE_SL, '--o', linewidth=3, markersize=12)
ax.loglog([3e1, 3e3], [2e-0, 2e-2], '--*k', linewidth=1, markersize=12)
ax.loglog([3e1, 3e3], [5e-2, 5e-3], '--k', linewidth=1, markersize=12)
ax.grid()
ax.legend(["MLMC", "MC", "-1", "-1/2"], fontsize=20)
ax.set_xlabel("Cost", fontsize=20)
ax.set_ylabel("MSE", fontsize=20)
ax.tick_params(axis='both', labelsize=15)
ax.tick_params(axis='x', rotation=45)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
# plt.savefig("1stepMSE{}R{}SLrate{}MLrate{}.eps".format(fun_name, R, rate_sl, rate_ml), format='eps')
plt.savefig("onestepMSE{}R{}MLandMCrate{:.4f}.eps".format(fun_name, R, rate_ml), format='eps')
# plt.show()




