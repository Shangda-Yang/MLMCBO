import time
import numpy as np
import torch
from botorch.test_functions import Hartmann, StyblinskiTang
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ObjectiveFunction import SelfDefinedFunction, Triangular1D, HalfCircle

from get_next_one import get_next_point_oneEI, get_next_point_onePI
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.manual_seed(99)

# target function
# target_fn = Hartmann(dim=6, negate=True)
# bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
# train_x = torch.rand(10, 6)
# train_y = target_fn(train_x).unsqueeze(-1)
# best_value = train_y.max()

num_obs = 6
target = SelfDefinedFunction()
bounds = target.bounds.to(dtype)
lower_bounds, upper_bounds = bounds[0].item(), bounds[1].item()
train_X = (upper_bounds-lower_bounds)*torch.rand(num_obs, 1, device=device, dtype=dtype) + lower_bounds
# train_X = torch.tensor([[-7.0], [-2.0], [0.0], [1.0]], device=device, dtype=dtype)
# train_X = torch.tensor([[-5.0], [5.0]], device=device, dtype=dtype)
train_Y = target(train_X).unsqueeze(-1)

# lower_bounds = -5.0
# upper_bounds = 5.0
# num_obs = 10
# bounds = torch.tensor([[lower_bounds] * 2, [upper_bounds] * 2], device=device, dtype=dtype)
#
# target = StyblinskiTang(negate=True)
# train_X = 10*torch.rand(num_obs, 2, device=device, dtype=dtype) - 5
# train_Y = target(train_X).unsqueeze(-1)

best = train_Y.max().item()

n_runs = 1

num_restarts = 20
raw_samples = 512

N_ref = 2**12

# X_fan = torch.linspace(-10, 10, N_ref, device=device, dtype=dtype)
# X_fan = X_fan.repeat(mesh_num, 1)
# X = torch.cat([X_zero, X_fan], 1).unsqueeze(-1)
# print(X.size())

# reference solution
train_x = train_X
train_y = train_Y
best_value = best

ref_candidate = torch.zeros((n_runs, 1))
ref_value = torch.zeros((n_runs, 1))

print('Start')

start_time = time.time()
for run in range(n_runs):
    aq_candidate, aq_value, _ = get_next_point_oneEI(train_x, train_y,
                                                     bounds,
                                                     num_samples=N_ref,
                                                     num_restarts=num_restarts,
                                                     raw_samples=raw_samples,
                                                     )
    new_results = target(aq_candidate).unsqueeze(-1)

    train_x = torch.cat([train_x, aq_candidate])
    train_y = torch.cat([train_y, new_results])

    best_value = train_y.max().item()
    print("--- {:.4f} seconds ---".format(time.time() - start_time))

    ref_candidate[run] = aq_candidate
    ref_value[run] = aq_value

# rate of convergence with respect to N
N = torch.tensor([2 ** power for power in range(1, 8)])
R = 100

error_point = torch.zeros(R, len(N), n_runs)
error_value = torch.zeros(R, len(N), n_runs)

for i in range(len(N)):
    start_time = time.time()
    n = N[i]
    for j in range(R):
        train_x = train_X
        train_y = train_Y
        best_value = best
        for run in range(n_runs):
            aq_candidate_app, aq_value_app, _ = get_next_point_oneEI(train_x, train_y,
                                                                     bounds,
                                                                     num_samples=n,
                                                                     num_restarts=num_restarts,
                                                                     raw_samples=raw_samples
                                                                     )
            new_results = target(aq_candidate_app).unsqueeze(-1)

            train_x = torch.cat([train_x, aq_candidate_app])
            train_y = torch.cat([train_y, new_results])

            best_value = train_y.max().item()

            error_point[j, i, run] = torch.norm(aq_candidate_app.cpu() - ref_candidate[run].cpu())
            error_value[j, i, run] = torch.norm(aq_value_app.cpu() - ref_value[run].cpu())
    print("--- N {:4d}, {:.4f} seconds ---".format(n, time.time() - start_time))

print("Run Finished")

# error
for i in range(n_runs):
    mse_point = torch.sum(error_point[:, :, i] ** 2, dim=0) / R
    mse_value = torch.sum(error_value[:, :, i] ** 2, dim=0) / R

    rate = np.polyfit(torch.log(N), torch.log(mse_point), 1)[0]
    print(rate)
    print(np.polyfit(torch.log(N), torch.log(mse_value), 1)[0])

    # plot
    # fig, (ax1) = plt.subplots(1, 1, figsize=(16, 9))
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
    fig.tight_layout(pad=10.0)

    ax1.loglog(N[1:], mse_point[1:], '--*', linewidth=3, markersize=12)
    ax1.grid()
    ax1.set_xlabel(r'$N$', fontsize=20)
    ax1.set_ylabel(r"$\mathbb{E}[||x_N^{*} - x^*||_2^2]$", fontsize=20)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    # ax2.loglog(N, mse_value, '--*', linewidth=3, markersize=12)
    # ax2.grid()
    # ax2.set_xlabel(r'$N$', fontsize=20)
    # ax2.set_ylabel(r"$\mathbb{E}[|\alpha_{N}(x_N^*) - \alpha(x^*)|^2]$", fontsize=20)
    # ax2.tick_params(axis='both', labelsize=15)
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # plt.savefig("../figures/OneStepRate/OneEIMaximizerStrongRate{:.4f}obs{}rels{}.eps".format(rate, num_obs, R), format='eps')
    plt.show()
