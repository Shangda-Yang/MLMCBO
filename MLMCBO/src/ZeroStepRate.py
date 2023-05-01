# %%
import math
import pickle
import time
import torch
from botorch.test_functions import Hartmann
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import linregress
import numpy as np
from get_next_zero import get_next_point_pi, get_next_point_ei
from ObjectiveFunction import SelfDefinedFunction

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.manual_seed(99)

# test function
# target = Hartmann(dim=6, negate=True)
# bounds = torch.tensor([[0.0]*6, [1.0]*6], device=device, dtype=dtype)
# train_X = torch.rand(10, 6, device=device, dtype=dtype)
# train_Y = target(train_X).unsqueeze(-1)
num_obs = 6
target = SelfDefinedFunction()
bounds = target.bounds
lower_bounds, upper_bounds = bounds[0].item(), bounds[1].item()
train_X = 2*upper_bounds*torch.rand(num_obs, 1, device=device, dtype=dtype) - upper_bounds
# train_X = torch.tensor([[-1.0], [1.0]], device=device, dtype=dtype)
train_Y = target(train_X).unsqueeze(-1)
best = train_Y.max().item()

X = torch.linspace(lower_bounds, upper_bounds, 1000, device=device, dtype=dtype).unsqueeze(-1)

# test codes
n_runs = 12

# reference solution
train_x = train_X
train_y = train_Y
best_value = best

ref_candidate = torch.zeros((n_runs, 1))
ref_value = torch.zeros((n_runs, 1))
# %%
#
# with open(".\\Data\\EIref_zero_step.pkl", 'wb') as f:
#     pickle.dump([EI_candidate, EI_value], f)

# %%
# MC PI
# with open("./Data/EIref_zero_step.pkl", 'rb') as f:
#     EI_candidate, EI_value = pickle.load(f)

N = torch.tensor([2 ** power for power in range(1, 12)])
# N = torch.tensor(list(range(2, 129, 2)))
R = 100

error_point = torch.zeros(R, len(N), n_runs)
error_value = torch.zeros(R, len(N), n_runs)

for run in range(n_runs):
    start_time = time.time()
    best_value = train_y.max().item()
    aq_candidate_ref, aq_value_ref, _, ac_func = get_next_point_ei(train_x, train_y,
                                                                   best_value, bounds,
                                                                   num_restarts=20,
                                                                   raw_samples=512,
                                                                   X=X)
    print("--------------------------------------------------------")
    print("--- Run {} {:.4f} seconds ---".format(run, time.time() - start_time))

    ref_candidate[run] = aq_candidate_ref
    ref_value[run] = aq_value_ref

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout(pad=10.0)
    ax.plot(X.cpu().detach().numpy(), ac_func, label='Analytic', color='k')
    ax.scatter(aq_candidate_ref.cpu(), aq_value_ref.cpu(), color='k')
    ax.grid()
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel(r"$\alpha(x)$", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.savefig("../figures/SAA/EIfunc{}obs{}obser.eps".format(num_obs, run))
    plt.show()
    for i in range(len(N)):
        n = N[i]
        start_time = time.time()
        for j in range(R):
            aq_candidate_app, aq_value_app, _ = get_next_point_ei(train_x, train_y,
                                                                  best_value, bounds,
                                                                  num_restarts=20,
                                                                  raw_samples=512,
                                                                  num_samples=n)
            error_point[j, i, run] = torch.norm(aq_candidate_app - ref_candidate[run])
            error_value[j, i, run] = torch.norm(aq_value_app - ref_value[run])
        print("--- N {:4d}, {:.4f} seconds ---".format(n, time.time() - start_time))


    mse_point = torch.sum(np.abs(error_point[:, :, run]), dim=0) / R
    mse_value = torch.sum(error_value[:, :, run] ** 2, dim=0) / R


    point_rate = np.polyfit(torch.log(N), torch.log(mse_point), 1)[0]
    print(point_rate)
    print(np.polyfit(torch.log(N), torch.log(mse_value.detach()), 1)[0])

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    fig.tight_layout(pad=10.0)

    ax1.loglog(N, mse_point, '--*', linewidth=3, markersize=12)
    ax1.grid()
    ax1.set_xlabel(r'$N$', fontsize=20)
    ax1.set_ylabel(r"$\mathbb{E}[||x_N^{*} - x^*||_2]$", fontsize=20)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.savefig("../figures/SAA/ZeroRate{:.4f}obs{}relsTracked.eps".format(point_rate, run), format='eps')
    plt.show()

    # fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    # ax2.loglog(N, mse_value.detach(), '--*', linewidth=3, markersize=12)
    # ax2.grid()
    # ax2.set_xlabel(r'$N$', fontsize=20)
    # ax2.set_ylabel(r"$\mathbb{E}[|\alpha_N(x_N^{*}) - \alpha(x^*)|^2]$", fontsize=20)
    # ax2.tick_params(axis='both', labelsize=15)
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # plt.savefig("../figures/SelfDefinedFunction/ZeroRate{}obs{}relsTracked.eps".format(num_obs, R), format='eps')
    # plt.show()

    new_results = target(aq_candidate_ref).unsqueeze(-1)

    train_x = torch.cat([train_x, aq_candidate_ref])
    train_y = torch.cat([train_y, new_results])

    best_value = train_y.max().item()

# for i in range(len(N)):
#
#
#     for j in range(R):
#         train_x = train_X
#         train_y = train_Y
#         best_value = best
#         for run in range(n_runs):
#             train_x = 2 * upper_bounds * torch.rand(num_obs, 1, device=device, dtype=dtype) - upper_bounds
#             # train_X = torch.tensor([[-5.0], [5.6], [6.2], [6.5], [9.0]], device=device, dtype=dtype)
#             train_y = target(train_x).unsqueeze(-1)
#             best = train_y.max().item()
#             aq_candidate, aq_value, _, ac_func = get_next_point_ei(train_x, train_y,
#                                                                    best_value, bounds,
#                                                                    num_restarts=20,
#                                                                    raw_samples=512,
#                                                                    X=X)
#             # aq_candidate, aq_value, _ = get_next_point_ei(train_x, train_y,
#             #                                               best_value, bounds,
#             #                                               num_restarts=10,
#             #                                               raw_samples=512)
#
#             # aq_candidate_app, aq_value_app, _, ac_func = get_next_point_ei(train_x, train_y,
#             #                                                       best_value, bounds,
#             #                                                       num_restarts=20,
#             #                                                       raw_samples=512,
#             #                                                       num_samples=n,
#             #                                                       X=X)
#             aq_candidate_app, aq_value_app, _ = get_next_point_ei(train_x, train_y,
#                                                                   best_value, bounds,
#                                                                   num_restarts=20,
#                                                                   raw_samples=512,
#                                                                   num_samples=n)
#             new_results = target(aq_candidate_app).unsqueeze(-1)
#
#             train_x = torch.cat([train_x, aq_candidate_app])
#             train_y = torch.cat([train_y, new_results])
#
#             best_value = train_y.max().item()

            # fig, ax = plt.subplots(figsize=(10, 8))
            # fig.tight_layout(pad=10.0)
            # ax.plot(X.cpu().detach().numpy(), ac_func, '--', zorder=0, label="N = {}".format(n))
            # ax.scatter(aq_candidate_app.detach(), aq_value_app.detach(), zorder=0)
            # # ax.grid()
            # ax.set_xlabel('x', fontsize=20)
            # ax.set_ylabel(r'$\alpha_N(x)$', fontsize=20)
            # ax.tick_params(axis='both', which='major', labelsize=15)
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
            # plt.show()



# ax.grid()
# plt.legend(loc=2)
# # plt.savefig("../figures/SelfDefinedFunction/EIfunc{}obser.eps".format(num_obs), format="eps")
# plt.show()
print("Run Finished")
# %%
# error

# for i in range(n_runs):
    # mse_point = torch.sum(np.abs(error_point[:, :, 1]), dim=0) / R
    # mse_value = torch.sum(error_value[:, :, 1] ** 2, dim=0) / R
    #
    # print(np.polyfit(torch.log(N), torch.log(mse_point), 1)[0])
    # print(np.polyfit(torch.log(N), torch.log(mse_value.detach()), 1)[0])
    #
    # # plot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    # fig.tight_layout(pad=10.0)
    #
    # ax1.loglog(N, mse_point, '--*', linewidth=3, markersize=12)
    # ax1.grid()
    # ax1.set_xlabel(r'$N$', fontsize=20)
    # ax1.set_ylabel(r"$\mathbb{E}[|x_N^{*} - x^*|^2]$", fontsize=20)
    # ax1.tick_params(axis='both', labelsize=15)
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #
    # ax2.loglog(N, mse_value.detach(), '--*', linewidth=3, markersize=12)
    # ax2.grid()
    # ax2.set_xlabel(r'$N$', fontsize=20)
    # ax2.set_ylabel(r"$\mathbb{E}[|\alpha_N(x_N^{*}) - \alpha(x^*)|^2]$", fontsize=20)
    # ax2.tick_params(axis='both', labelsize=15)
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # # plt.savefig("../figures/SelfDefinedFunction/ZeroRate{}obs{}relsTracked.eps".format(num_obs, R), format='eps')
    # plt.show()
