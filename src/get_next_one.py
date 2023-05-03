import torch

from OneStep import OneStepEI, OneStepPI
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from GPModels import GPmll

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.double


# ----------------------------------------------------------------------- #
# Expected improvement
def get_next_point_ei(train_x, train_y, best_value, bounds,
                      num_restarts=20, raw_samples=512, num_samples=0, X=None):
    model = GPmll(train_x, train_y)

    if num_samples == 0:
        EI = ExpectedImprovement(model=model, best_f=best_value)
        new_candidate, new_value = optimize_acqf(acq_function=EI,
                                                 bounds=bounds,
                                                 q=1,
                                                 num_restarts=num_restarts,
                                                 raw_samples=raw_samples)

        if X is not None:
            ei_func = [EI(x.unsqueeze(-1)).item() for x in X]
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

# ------------------------------------------------------------------------------------ #
# One-step EI
# ------------------------------------------------------------------------------------ #
def get_next_point_oneEI(train_x, train_y, bounds, num_samples,
                         num_restarts=20, raw_samples=512, X=None,
                         num_samples_inner=None):
    model = GPmll(train_x, train_y)

    sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
    if num_samples_inner:
        inner_sampler = IIDNormalSampler(num_samples=num_samples_inner, resample=False)
    else:
        inner_sampler = None
    oneEI = OneStepEI(model=model,
                      num_fantasies=None,
                      sampler=sampler,
                      inner_sampler=inner_sampler)
    new_candidate, new_value = optimize_acqf(acq_function=oneEI,
                                             bounds=bounds,
                                             q=1,
                                             num_restarts=num_restarts,
                                             raw_samples=raw_samples)

    if X is None:
        return new_candidate, new_value, model
    else:
        oneEI.evaluate(X=X, bounds=bounds)
        ac_func = [oneEI.evaluate(X=x.unsqueeze(0), bounds=bounds).item() for x in X]
        return new_candidate, new_value, model, torch.tensor(ac_func)

# ------------------------------------------------------------------------------------ #
# One-step PI
# ------------------------------------------------------------------------------------ #
def get_next_point_onePI(train_x, train_y, bounds, num_samples,
                         num_restarts=20, raw_samples=512, X=None):
    model, mll = GPmll(train_x, train_y)

    sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
    onePI = OneStepPI(model=model,
                      num_fantasies=None,
                      sampler=sampler)
    new_candidate, new_value = optimize_acqf(acq_function=onePI,
                                             bounds=bounds,
                                             q=1,
                                             num_restarts=num_restarts,
                                             raw_samples=raw_samples)
    if X is None:
        return new_candidate, new_value, model
    else:
        ac_func = onePI(X)
        return new_candidate, new_value, model, ac_func
# ------------------------------------------------------------------------------------ #
