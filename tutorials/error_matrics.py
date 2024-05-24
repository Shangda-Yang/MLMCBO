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