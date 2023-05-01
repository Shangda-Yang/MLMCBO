import time
import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.test_functions import Hartmann, StyblinskiTang
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from ObjectiveFunction import SelfDefinedFunction, Triangular1D, HalfCircle
from get_next_one import get_next_point_oneEI, get_next_point_onePI
import pickle
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
train_X = 2*upper_bounds*torch.rand(num_obs, 1, device=device, dtype=dtype) - upper_bounds
# train_X = torch.tensor([[-7.0], [-2.0], [0.0], [1.0]], device=device, dtype=dtype)
# train_X = torch.tensor([[-1.0], [0.6], [0.7]], device=device, dtype=dtype)
train_Y = target(train_X).unsqueeze(-1)
best = train_Y.max().item()

# num_obs = 5
# target = StyblinskiTang()
# bounds = torch.tensor([[-5.0] * 2, [5.0] * 2], device=device, dtype=dtype)
# train_X = 10*torch.rand(num_obs, 2, device=device, dtype=dtype) - 5
# train_Y = target(train_X).unsqueeze(-1)
# best = train_Y.max().item()

n_runs = 1

num_restarts = 20
raw_samples = 512

N_ref = 2**5

# reference solution
train_x = train_X
train_y = train_Y
best_value = best

ref_candidate = torch.zeros((n_runs, 1))
ref_value = torch.zeros((n_runs, 1))

print('Start')

start_time = time.time()

t = 10
for run in range(n_runs):
    for i in range(t):
        aq_candidate, aq_value, _ = get_next_point_oneEI(train_x, train_y,
                                                         best_value, bounds,
                                                         num_samples=N_ref,
                                                         num_restarts=num_restarts,
                                                         raw_samples=raw_samples,
                                                         )
        # new_results = target(aq_candidate).unsqueeze(-1)
        #
        # train_x = torch.cat([train_x, aq_candidate])
        # train_y = torch.cat([train_y, new_results])
        #
        # best_value = train_y.max().item()
        print("--- {:.4f} seconds ---".format(time.time() - start_time))
        # print(aq_candidate)

        ref_candidate[run] = torch.add(ref_candidate[run], aq_candidate/t)
        ref_value[run] = torch.add(ref_value[run], aq_value/t)

# rate of convergence with respect to N
M = torch.tensor([2 ** power for power in range(1, 8)])
R = 100

points = torch.zeros(R, len(M), n_runs)
values = torch.zeros(R, len(M), n_runs)

for i in range(len(M)):
    start_time = time.time()
    m = M[i]
    for j in range(R):
        train_x = train_X
        train_y = train_Y
        best_value = best
        for run in range(n_runs):
            aq_candidate_app, aq_value_app, _ = get_next_point_oneEI(train_x, train_y,
                                                                     best_value, bounds,
                                                                     num_samples=N_ref,
                                                                     num_restarts=num_restarts,
                                                                     raw_samples=raw_samples,
                                                                     num_samples_inner=m,
                                                                     )
            # print(aq_candidate_app)
            new_results = target(aq_candidate_app).unsqueeze(-1)

            train_x = torch.cat([train_x, aq_candidate_app])
            train_y = torch.cat([train_y, new_results])

            best_value = train_y.max().item()

            points[j, i, run] = aq_candidate_app.cpu()
            values[j, i, run] = aq_value_app.cpu()
    print("--- N {:4d}, {:.4f} seconds ---".format(m, time.time() - start_time))

print("Run Finished")

# error
for i in range(n_runs):
    error_point = points[:, :, i] - ref_candidate[i].cpu()
    error_value = values[:, :, i] - ref_value[i].cpu()
    # mse_point = torch.norm(error_point, p=2, dim=0) ** 2
    # mse_value = torch.norm(error_value, p=2, dim=0) ** 2

    mse_point = torch.sum(error_point ** 2, dim=0) / R
    mse_value = torch.sum(error_value ** 2, dim=0) / R

    rate = np.polyfit(torch.log(M), torch.log(mse_point), 1)[0]
    print(rate)

    # plot
    # fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    fig.tight_layout(pad=10.0)

    ax1.loglog(M, mse_point, '--*', linewidth=3, markersize=12)
    ax1.grid()
    ax1.set_xlabel(r'$M$', fontsize=20)
    ax1.set_ylabel(r"$\mathbb{E}[||x_{N,M}^* - x_N^*||_2^2]$", fontsize=20)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    # ax2.loglog(M, mse_value, '--*', linewidth=3, markersize=12)
    # ax2.grid()
    # ax2.set_xlabel(r'$M$', fontsize=20)
    # ax2.set_ylabel(r"$\mathbb{E}[|\alpha_{N,M}(x_{N,M}^*) - \alpha_N(x_N^*)|^2]$", fontsize=20)
    # ax2.tick_params(axis='both', labelsize=15)
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.savefig("../figures/SelfDefinedFunction/NestedOneEIMaximizer{}Rate{}obs{}N{}rels{}starts.eps".format(rate, num_obs, N_ref, R, num_restarts), format='eps')
    plt.show()

# filename = os.path.abspath(os.path.join(__file__, "..", "..", 'Data', 'StyblinskiTang', 'NestedRate.pkl'))
# with open(filename, 'w') as f:
#     pickle.dump(train_x, train_y, error_value, error_point, mse_value, mse_point, f)
