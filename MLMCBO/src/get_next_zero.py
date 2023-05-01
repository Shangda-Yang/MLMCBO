import torch
from botorch import fit_gpytorch_model, fit_gpytorch_mll
from botorch.acquisition import ProbabilityOfImprovement, qProbabilityOfImprovement, UpperConfidenceBound, \
    qUpperConfidenceBound
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from GPModels import GPmll

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.double


# ----------------------------------------------------------------------- #
# Probability of improvement
# ----------------------------------------------------------------------- #
def get_next_point_pi(train_x, train_y, best_value, bounds,
                      num_restarts=20, raw_samples=512, num_samples=0, X=None):
    model = GPmll(train_x, train_y)
    # analytic solution
    if num_samples == 0:
        PI = ProbabilityOfImprovement(model=model, best_f=best_value)
        new_candidate, new_value = optimize_acqf(acq_function=PI,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples)
        if X is not None:
            pi_func = [PI(x.unsqueeze(-1)).item() for x in X]
    # approximation
    else:
        sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
        qPI = qProbabilityOfImprovement(model=model, best_f=best_value,
                                        sampler=sampler)
        new_candidate, new_value = optimize_acqf(acq_function=qPI,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples)
        if X is not None:
            pi_func = [qPI(x.unsqueeze(-1)).item() for x in X]
    if X is not None:
        return new_candidate, new_value, model, pi_func
    else:
        return new_candidate, new_value, model


# ----------------------------------------------------------------------- #
# Expected improvement
# ----------------------------------------------------------------------- #
def get_next_point_ei(train_x, train_y, best_value, bounds,
                      num_restarts=20, raw_samples=512, num_samples=0, X=None):
    model = GPmll(train_x, train_y)
    # analytic solution
    if num_samples == 0:
        EI = ExpectedImprovement(model=model, best_f=best_value)
        new_candidate, new_value = optimize_acqf(acq_function=EI,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples)

        if X is not None:
            ei_func = [EI(x.unsqueeze(-1)).item() for x in X]
    # approximation
    else:
        sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
        qEI = qExpectedImprovement(model=model, best_f=best_value,
                                   sampler=sampler)
        new_candidate, new_value = optimize_acqf(acq_function=qEI,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples,
                                                 return_best_only=True)
        if X is not None:
            ei_func = [qEI(x.unsqueeze(-1)).item() for x in X]
    if X is not None:
        return new_candidate, new_value, model, ei_func
    else:
        return new_candidate, new_value, model


# ----------------------------------------------------------------------- #
# Upper confidence bound
# ----------------------------------------------------------------------- #
def get_next_point_ucb(train_x, train_y, best_value, bounds,
                       num_restarts=20, raw_samples=512, num_samples=0):
    model = GPmll(train_x, train_y)
    # analytic solution
    if num_samples == 0:
        UCB = UpperConfidenceBound(model=model, beta=0.2)
        new_candidate, new_value = optimize_acqf(acq_function=UCB,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples)
    # approximation
    else:
        sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
        qUCB = qUpperConfidenceBound(model=model, best_f=best_value,
                                     sampler=sampler)
        new_candidate, new_value = optimize_acqf(acq_function=qUCB,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples)
    return new_candidate, new_value, model
# ------------------------------------------------------------------------------------ #
