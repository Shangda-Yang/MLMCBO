import time

import numpy as np


def optimize_mlmc(inc_function, eps, dl, alpha=1, beta=1.5, gamma=1, meanc=1, varc=1, var0=1):
    r"""Optimize acquisition functions using mlmc.

    Args:
        inc_function: define the acquisition function estimation at different levels
                      (single level estimator at l = 0 and increment otherwise)
        eps: desired accuracy
        dl: actual starting level
        alpha: bias rate
        beta: variance rate
        gamma: cost rate
        meanc: bias constant
        varc: variance constant for increment
        var0: variance for single level estimator

    Returns:
        a tuple of optimiser, corresponding optimum, number of samples applied
    """
    # TODO: Allow pre-computation of constants

    # total levels required (at least two levels)
    L = np.maximum(np.ceil(np.log2(np.sqrt(2) * meanc / (eps * (2 ** alpha - 1))) / alpha).astype(int), dl + 1)
    levels = np.arange(dl, L + 1)

    Ncon = (np.sum(np.sqrt(varc * 2 ** (-levels * (beta - gamma)))) + np.sqrt(var0 * 2 ** gamma)) / (eps ** 2)
    # number of outer samples at each level
    Nl = np.ceil(np.maximum(Ncon * np.sqrt(varc * 2 ** (-levels * (beta + gamma))), 2)).astype(int)

    # TODO: Check one-shot ml f is the same as single f(z^*)

    # z is the optimizer and f is the corresponding optimum
    z, f = 0, 0
    # matching optimizer such that the optimizers at levels correspond
    match = None
    for level, n in zip(levels - dl, Nl):
        zl, fl, match = inc_function.sample_candidate(level, dl, n, match=match)
        f += fl
        z += zl
    return z, f, Nl


