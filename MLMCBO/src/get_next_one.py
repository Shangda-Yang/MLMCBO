import torch
from OneStep import OneStepEI, OneStepPI
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from GPModels import GPmll

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.double
# ------------------------------------------------------------------------------------ #
# One-step EI
# ------------------------------------------------------------------------------------ #
def get_next_point_oneEI(train_x, train_y, best_value, bounds, num_samples,
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
                      inner_sampler=inner_sampler,
                      current_value=best_value)
    new_candidate, new_value = optimize_acqf(acq_function=oneEI,
                                             bounds=bounds,
                                             q=1,
                                             num_restarts=num_restarts,
                                             raw_samples=raw_samples,
                                             return_best_only=True)

    if X is None:
        return new_candidate, new_value, model
    else:
        oneEI.evaluate(X=X, bounds=bounds)
        ac_func = [oneEI.evaluate(X=x.unsqueeze(0), bounds=bounds).item() for x in X]
        return new_candidate, new_value, model, torch.tensor(ac_func)

# ------------------------------------------------------------------------------------ #
# One-step PI
# ------------------------------------------------------------------------------------ #
def get_next_point_onePI(train_x, train_y, best_value, bounds, num_samples,
                         num_restarts=20, raw_samples=512, X=None):
    model, mll = GPmll(train_x, train_y)

    sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
    onePI = OneStepPI(model=model,
                      num_fantasies=None,
                      sampler=sampler,
                      current_value=best_value)
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
