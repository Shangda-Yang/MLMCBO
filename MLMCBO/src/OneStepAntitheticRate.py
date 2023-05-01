import time
import numpy as np
import torch
from botorch.test_functions import Hartmann, StyblinskiTang
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from ObjectiveFunction import SelfDefinedFunction, Triangular1D, HalfCircle

from get_next_one_inc import get_next_point_oneEI_inc, get_next_point_oneEI_ant_inc
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.manual_seed(99)

num_obs = 6
target = SelfDefinedFunction()
bounds = target.bounds.to(dtype)
lower_bounds, upper_bounds = bounds[0].item(), bounds[1].item()
train_X = 2*upper_bounds*torch.rand(num_obs, 1, device=device, dtype=dtype) - upper_bounds
train_Y = target(train_X).unsqueeze(-1)

best = train_Y.max().item()

n_runs = 1

num_restarts = 20
raw_samples = 512

N_ref = 2**5

print('Start')

start_time = time.time()
# rate of convergence with respect to N
L = range(4, 9)
M = torch.tensor([2 ** power for power in L])
R = 100

error_point = torch.zeros(R, len(M), n_runs)

for i in range(len(M)):
    start_time = time.time()
    m = M[i]
    for j in range(R):
        train_x = train_X
        train_y = train_Y
        best_value = best
        for run in range(n_runs):
            ant_inc = get_next_point_oneEI_ant_inc(train_x, train_y,
                                                   best_value, bounds,
                                                   num_samples=N_ref,
                                                   num_restarts=num_restarts,
                                                   raw_samples=raw_samples,
                                                   num_samples_inner=m,
                                                   )
            new_results = target(ant_inc).unsqueeze(-1)

            error_point[j, i, run] = ant_inc.cpu()
        print("Sample {}, Realisation {}".format(m, j))
    print("--- M {:4d}, {:.4f} seconds ---".format(m, time.time() - start_time))

print("Run Finished")

# error
for i in range(n_runs):
    var_point = N_ref * torch.sum(error_point[:, :, i] ** 2, dim=0) / R

    rate = np.polyfit(L, torch.log2(var_point), 1)[0]
    print(rate)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.tight_layout(pad=10.0)

    ax.plot(L, torch.log2(var_point), '--*', linewidth=3, markersize=12)
    ax.grid()
    ax.set_xlabel(r'$l$', fontsize=20)
    ax.set_ylabel(r"$\log_{2}\mathbb{E}[|\Delta x_l^*|^2]$", fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    plt.savefig("../figures/SelfDefinedFunction/AntIncEIrate{}obs{}relsBeta{}.eps".format(num_obs, R, rate), format='eps')
    plt.show()
