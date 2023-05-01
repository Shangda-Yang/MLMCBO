import numpy as np
from get_next_multistep import get_next_point_twoEI


# --------------------------------------------------------------------------- #
def mc_twoEI(eps, alpha, target, **kwargs):
    r"""
    Monte Carlo routine with two-step lookahead EI
    - run Monte Carlo simulation with given accuracy (eps)
    Parameters
    ------
    eps:        required accuracy
    alpha:      convergence rate of mean
    target:     objective function
    kwargs
    """
    train_x = kwargs["train_x"]
    train_y = kwargs["train_y"]
    bounds = kwargs["bounds"]
    num_restarts = kwargs["num_restarts"]
    raw_samples = kwargs["raw_samples"]

    L = np.round(-np.log2(eps**2)/alpha).astype(int)
    N = np.round((1/np.power(eps, 2))).astype(int)

    M = np.max((2**L, 2))

    new_candidate, _ = get_next_point_twoEI(train_x,
                                            train_y,
                                            bounds,
                                            num_samples=[N, M],
                                            num_restarts=num_restarts,
                                            raw_samples=raw_samples
                                            )
    cost = N*M
    # print(aq_candidate_app)
    new_result = target(new_candidate).unsqueeze(-1)

    return new_candidate, new_result, cost
