import time
import numpy as np
import torch
from botorch.test_functions import Rastrigin
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mlmc_run import mlmc_run
from inc_fn import test_fn_ant_inc
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.manual_seed(66)
dim = 4
num_obs = 8
target = Rastrigin(dim=dim)
fun_name = "Rastrigin"
bounds = target.bounds
lower_bounds, upper_bounds = bounds[0], bounds[1]
train_x = 2*upper_bounds*torch.rand(num_obs, 1, device=device, dtype=dtype) - upper_bounds
train_y = target(train_x).unsqueeze(-1)

n_runs = 10

num_restarts = 20
raw_samples = 256

Eps = [0.5]

R = 1

alpha = 1
beta = 1.5
gamma = 1

kwargs = {"train_x": train_x,
          "train_y": train_y,
          "bounds": bounds,
          "num_restarts": num_restarts,
          "raw_samples": raw_samples
          }

results = torch.zeros(R, len(Eps))
candidates = torch.zeros(R, len(Eps), dim)
costs = torch.zeros(R, len(Eps))

for i in range(len(Eps)):
    eps = Eps[i]
    for j in range(R):
        best_result, best_candidate, Cl = mlmc_run(eps, test_fn_ant_inc, alpha, beta, gamma, n_runs, target, **kwargs)
        results[j, i] = best_result
        candidates[j, i, :] = best_candidate
        costs[j, i] = np.sum(Cl)

error = torch.norm(results - target.optimal_value, 2).cpu()
cost = torch.sum(costs, dim=0) / R
rate = np.polyfit(cost, error**2, 1)[0]
print(rate)

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.tight_layout(pad=10.0)

ax.plot(cost, error, '--*', linewidth=3, markersize=12)
ax.grid()
ax.set_xlabel("Cost", fontsize=20)
ax.set_ylabel("MSE", fontsize=20)
ax.tick_params(axis='both', labelsize=15)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

plt.savefig("../figures/MLMC/MLMC{}R{}Rate{}.eps".format(fun_name, R, rate), format='eps')
plt.show()




