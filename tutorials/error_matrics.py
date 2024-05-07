import numpy as np
import pandas as pd
import torch


def compute_nmse(results, costs, reference, relative, R):
    # compute the NMSE, error bar, cumulative costs
    error = (results.cpu() - reference)
    MSE = torch.sum(error ** 2 / relative, dim=0) / R
    errorBar = 1.96 * torch.std(error ** 2 / relative, dim=0) / np.sqrt(len(MSE))
    Cost = torch.mean(torch.cumsum(costs, dim=1), dim=0)
    return MSE, errorBar, Cost


def compute_regret(results, ref, fun_name, num_obs, n_runs, R, ML=True):
    if ML is True:
        ml = 'MLMC'
    else:
        ml = 'MC'
    error = ref - results.cpu()
    regret = torch.mean(error, dim=0)
    errorBar = 1.645 * torch.std(error, dim=0) / np.sqrt(len(regret))

    errorBar[errorBar < 0] = 0

    regret_df = pd.DataFrame(regret.numpy())
    regret_df.to_csv('./Data/Regret{}Init{}Evals{}R{}{}.csv'.format(fun_name, num_obs, n_runs, R, ml))

    errorBar_np = errorBar.numpy()
    errorBar_df = pd.DataFrame(errorBar_np)
    errorBar_df.to_csv('./Data/ErrorBar{}Init{}Evals{}R{}{}.csv'.format(fun_name, num_obs, n_runs, R, ml))

    results_np = results.numpy()
    results_df = pd.DataFrame(results_np)
    results_df.to_csv('./Data/Results{}Init{}Evals{}R{}{}.csv'.format(fun_name, num_obs, n_runs, R, ml))
    return regret, errorBar


def GAP(results, rel, ref, fun_name):
    gap = (results - rel) / (ref - rel)
    mean_gap = torch.mean(gap, dim=0)
    median_gap = torch.median(gap, dim=0).values
    with open('./Data/{}Matern.txt'.format(fun_name), 'w') as file:
        file.write("ML GAP Mean: {:.5f}, GAP Median: {:.5f}".format(mean_gap[-1], median_gap[-1]))
    return mean_gap, median_gap
