import time

import numpy as np

def mlmc(eps, inc_fn, dl, alpha, beta, gamma, meanc, varc, var0, **kwargs):
    """
    Multilevel Monte Carlo estimation.
    The function will calculate the level and
    the number of samples at each level automatically.

    Parameters
    ------
    eps:    desired accuracy
    inc_fn: the function used to calculate the increments
    alpha:  weak error is O(2^{-alpha*l})
    beta:   strong error is O(2^{-beta*l})
    gamma:  cost of single sample is O(2^{gamma*l})
    meanc:  constant of mean convergence
    varc:   constant of variance convergence
    var0:   variance at level 0
    kwargs

    Returns
    ------
        a tuple (f, Nl, Cl)
        f: mlmc approximation
        Nl: number of samples at each level
        Cl: cost of samples at each level
    """
    L = np.ceil(1*np.log2(np.sqrt(2)*meanc/(eps*(2**alpha-1))) / alpha).astype(int)
    L = max(L, dl + 1)
    Ncon = (np.sum([np.sqrt(varc * 2 ** (-x*(beta-gamma))) for x in range(dl, L + 1)])
            + np.sqrt(var0 * 2**gamma)) / (eps**2)
    Nl = np.ceil(np.array([max(1*Ncon * np.sqrt(varc * 2 ** (-x*(beta+gamma))), 2) for x in range(dl, L + 1)])).astype(int)

    f = 0.0
    Cl = np.zeros(L - dl + 1)

    train_x = kwargs["train_x"]
    train_y = kwargs["train_y"]
    num_restarts = kwargs["num_restarts"]
    raw_samples = kwargs["raw_samples"]
    bounds = kwargs["bounds"]

    for l in range(L - dl):
        # start_time = time.time()
        n = Nl[l]
        fl = inc_fn(train_x, train_y, l, dl, n, num_restarts, raw_samples, bounds)
        f += fl

        # costl = time.time() - start_time
        # Cl[l] = round(costl, 2)
        Cl[l] = 2 * n * 2 ** (l + dl)

    return (f, Nl, Cl)