import numpy as np
from mlmc import mlmc
from constants import mlmc_constants

# --------------------------------------------------------------------------- #
# Multilevel Monte Carlo routine with increment function (inc_fn)
# - run Monte Carlo simulation with given accuracy (eps)
# --------------------------------------------------------------------------- #
def mlmc_run(eps, inc_fn, dl, alpha, beta, gamma, target, **kwargs):
    r"""
    Multilevel Monte Carlo routine with increment function (inc_fn)
    - run Monte Carlo simulation with given accuracy (eps)

    Parameters
    ------
    eps:    desired accuracy
    inc_fn: the function used to calculate the increments
    dl:     starting level
    alpha:  weak error is O(2^{-alpha*l})
    beta:   strong error is O(2^{-beta*l})
    gamma:  cost of single sample is O(2^{gamma*l})
    target: objective function
    kwargs

    Returns
    ------
        a tuple (f, Nl, Cl)
        f: mlmc approximation
        Nl: number of samples at each level
        Cl: cost of samples at each level
    """
    R = 5
    Lmax = dl + 3
    Lmin = dl
    dN = 5
    # empirical mean and variance for computing number of samples and finest level
    meanr, meanc, var0, varr, varc = mlmc_constants(inc_fn, R, Lmin, Lmax, dN, **kwargs)
    # meanc, var0, varc = 1, 1, 1
    if alpha is None and beta is None and gamma is None:
        alpha = meanr
        beta = varr
        gamma = 1
    # meanc, varc, var0 = 1, 1, 1
    new_candidate, _, Cl = mlmc(eps, inc_fn, dl, alpha, beta, gamma, meanc, varc, var0, **kwargs)
    new_result = target(new_candidate).unsqueeze(-1)

    return new_candidate, new_result, Cl
