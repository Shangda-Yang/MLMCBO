import numpy as np
from get_next_one import get_next_point_oneEI


# --------------------------------------------------------------------------- #
def mc_run(eps, alpha, target, ref=False, **kwargs):
    r"""
    Monte Carlo routine with one-step lookahead EI
    - run Monte Carlo simulation with given accuracy (eps)
    Parameters
    ------
    eps:        required accuracy
    alpha:      convergence rate of mean
    target:     objective function
    ref:        if for reference solution
    kwargs
    """
    train_x = kwargs["train_x"]
    train_y = kwargs["train_y"]
    bounds = kwargs["bounds"]
    num_restarts = kwargs["num_restarts"]
    raw_samples = kwargs["raw_samples"]

    L = np.round(-np.log2(eps**2)/alpha).astype(int)
    N = np.round((1/np.power(eps, 2))).astype(int)
    # N = eps
    if ref == True:
        M = None
    else:
        M = np.max((2**L, 2))
        # M = N
    # new_candidate = torch.tensor([0.0])
    # while (torch.abs(new_candidate - 2.7) > 1):
    new_candidate, _, _ = get_next_point_oneEI(train_x,
                                               train_y,
                                               bounds,
                                               num_samples=N,
                                               num_restarts=num_restarts,
                                               raw_samples=raw_samples,
                                               num_samples_inner=M,
                                               )
    if ref == False:
        cost = N*M
    else:
        cost = N
    # print(aq_candidate_app)
    new_result = target(new_candidate).unsqueeze(-1)
    return new_candidate, new_result, cost
# --------------------------------------------------------------------------- #
