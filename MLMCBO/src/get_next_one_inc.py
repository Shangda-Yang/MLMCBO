import torch
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from OneStepIncrement import OneStepIncEI, OneStepIncAntEI
from GPModels import GPmll

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.double

# ------------------------------------------------------------------------------------- #
# One-step EI
# ------------------------------------------------------------------------------------- #
def get_next_point_oneEI_inc(train_x, train_y, best_value, bounds, num_samples,
                             num_restarts=20, raw_samples=512, num_samples_inner=None):
    model = GPmll(train_x, train_y)

    seed_out = torch.randint(0, 1000000, (1,)).item()
    seed_in = torch.randint(0, 1000000, (1,)).item()
    sampler = IIDNormalSampler(num_samples=num_samples, resample=False, seed=seed_out)
    inner_sampler = IIDNormalSampler(num_samples=num_samples_inner, resample=False, seed=seed_in)
    oneEI_f = OneStepIncEI(model=model,
                           num_fantasies=None,
                           sampler=sampler,
                           inner_sampler=inner_sampler,
                           current_value=best_value,
                           fc=1,
                           bounds=bounds,
                           num_restarts=num_restarts,
                           raw_samples=raw_samples
                           )

    new_candidate_f, _ = optimize_acqf(acq_function=oneEI_f,
                                       bounds=bounds,
                                       q=1,
                                       num_restarts=num_restarts,)

    sampler = IIDNormalSampler(num_samples=num_samples, resample=False, seed=seed_out)
    inner_sampler = IIDNormalSampler(num_samples=num_samples_inner, resample=False, seed=seed_in)
    oneEI_c = OneStepIncEI(model=model,
                           num_fantasies=None,
                           sampler=sampler,
                           inner_sampler=inner_sampler,
                           current_value=best_value,
                           fc=1,
                           bounds=bounds,
                           num_restarts=num_restarts,
                           raw_samples=raw_samples
                           )

    new_candidate_c, _ = optimize_acqf(acq_function=oneEI_c,
                                       bounds=bounds,
                                       q=1,
                                       num_restarts=num_restarts,
                                       raw_samples=raw_samples)

    new_candidate = new_candidate_f - new_candidate_c
    return new_candidate


# ------------------------------------------------------------------------------------ #
# One-step antithetic EI
# ------------------------------------------------------------------------------------- #
def get_next_point_oneEI_ant_inc(train_x, train_y, best_value, bounds, num_samples,
                                 num_restarts=20, raw_samples=512, num_samples_inner=None):
    model = GPmll(train_x, train_y)

    seed_out = torch.randint(0, 1000000, (1,)).item()
    seed_in = torch.randint(0, 1000000, (1,)).item()
    sampler = IIDNormalSampler(num_samples=num_samples, resample=False, seed=seed_out)
    inner_sampler = IIDNormalSampler(num_samples=num_samples_inner, resample=False, seed=seed_in)
    oneEI_f = OneStepIncAntEI(model=model,
                              num_fantasies=None,
                              sampler=sampler,
                              inner_sampler=inner_sampler,
                              current_value=best_value,
                              fc=0,
                              bounds=bounds,
                              num_restarts=num_restarts,
                              raw_samples=raw_samples
                              )

    new_candidate_f, _ = optimize_acqf(acq_function=oneEI_f,
                                       bounds=bounds,
                                       q=1,
                                       num_restarts=num_restarts,
                                       raw_samples=raw_samples)

    sampler = IIDNormalSampler(num_samples=num_samples, resample=False, seed=seed_out)
    inner_sampler = IIDNormalSampler(num_samples=num_samples_inner, resample=False, seed=seed_in)
    oneEI_c = OneStepIncAntEI(model=model,
                              num_fantasies=None,
                              sampler=sampler,
                              inner_sampler=inner_sampler,
                              current_value=best_value,
                              fc=1,
                              bounds=bounds,
                              num_restarts=num_restarts,
                              raw_samples=raw_samples
                              )

    new_candidate_c, _ = optimize_acqf(acq_function=oneEI_c,
                                       bounds=bounds,
                                       q=1,
                                       num_restarts=num_restarts,
                                       raw_samples=raw_samples)
    new_candidate = new_candidate_f - new_candidate_c
    return new_candidate
# ------------------------------------------------------------------------------------ #
