from get_next_zero import get_next_point_pi, get_next_point_ei

def ei_run(target, **kwargs):
    r"""To run expected improvement
    Parameters
    ------
    target: objective function
    **kwargs:       train_x, train_y, num_restarts, raw_samples, bounds

    Returns
    ------
    aq_candidate:   next candidate (observation point)
    new_result:     corresponding observation
    """
    train_x = kwargs["train_x"]
    train_y = kwargs["train_y"]
    bounds = kwargs["bounds"]
    num_restarts = kwargs["num_restarts"]
    raw_samples = kwargs["raw_samples"]
    best_value = train_y.max().item()
    aq_candidate, aq_value, _ = get_next_point_ei(train_x, train_y,
                                                  best_value, bounds,
                                                  num_restarts=num_restarts,
                                                  raw_samples=raw_samples,
                                                  )
    new_result = target(aq_candidate).unsqueeze(-1)
    return aq_candidate, new_result
