import numpy as np
import torch

# ---------------------------------------------------------------- #
# Compute the empirical mean and variance for MLMC
# ---------------------------------------------------------------- #
def mlmc_constants(inc_fn, R, Lmin, Lmax, num_samples, **kwargs):
    r"""
    Parameters
    ------
    inc_fn:         function of increment
    R:              number of realizations
    Lmin:           finest level
    Lmax:           coarsest level
    num_samples:    number of samples to approximate increment
    **kwargs:       train_x, train_y, num_restarts, raw_samples, bounds

    Returns
    -------
    meanr: convergence rate of mean
    meanc: constant of mean convergence
    var0:  variance at level 0
    varr:  convergence rate of variance
    varc:  constant of variance convergence
    """

    train_x = kwargs["train_x"]
    train_y = kwargs["train_y"]
    num_restarts = kwargs["num_restarts"]
    raw_samples = kwargs["raw_samples"]
    bounds = kwargs["bounds"]

    means = np.zeros(Lmax - Lmin + 1)
    vars = np.zeros(Lmax - Lmin + 1)
    for l in range(Lmax - Lmin + 1):
        for i in range(R):
            inc = inc_fn(train_x, train_y, l, Lmin, num_samples,
                             num_restarts, raw_samples, bounds)
            means[l - Lmin] += torch.norm(inc, p=1).item()
            vars[l - Lmin] += torch.norm(inc, p=2).item()**2

    means /= R
    vars /= R
    meanr, meanc = np.polyfit(range(Lmin + 1, Lmax + 1), np.log2(means[1:]), 1)
    varr, varc = np.polyfit(range(Lmin + 1, Lmax + 1), np.log2(vars[1:]), 1)
    var0 = vars[0]

    return meanr, meanc, var0, varr, varc





