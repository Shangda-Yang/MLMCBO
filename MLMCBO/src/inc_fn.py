import torch
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from GPmodels import GPmll

from onestep_inc import OneStepIncAntEI
from twostep_inc import TwoStepIncAntEI

# ---------------------------------------------------------------------------- #
def test_fn_ant_inc(train_x, train_y, l, dl, num_samples, num_restarts, raw_samples, bounds):
    r"""
    Antithetic increment 1-step EI

    Parameters
    ------
    train_x
    train_y
    l:          dl + l is the level
    dl:         starting level
    """
    M = 2 ** (l + dl)
    model = GPmll(train_x, train_y)
    if l == 0:
        sampler = IIDNormalSampler(num_samples=num_samples, resample=False)
        inner_sampler = IIDNormalSampler(num_samples=M, resample=False)
        oneEI = OneStepIncAntEI(model=model,
                                sampler=sampler,
                                inner_sampler=inner_sampler,
                                fc=0,
                                bounds=bounds)

        new_candidate, _ = optimize_acqf(acq_function=oneEI,
                                         bounds=bounds,
                                         q=1,
                                         num_restarts=num_restarts,
                                         raw_samples=raw_samples)
    else:
        seed = torch.randint(0, 10000000, (1,)).item()
        seed_out = torch.randint(0, 10000000, (1,)).item()
        sampler = IIDNormalSampler(num_samples=num_samples, resample=False, seed=seed_out)
        inner_sampler = IIDNormalSampler(num_samples=M, resample=False, seed=seed)
        oneEI_f = OneStepIncAntEI(model=model,
                                  sampler=sampler,
                                  inner_sampler=inner_sampler,
                                  fc=0,
                                  bounds=bounds)

        new_candidate_f, _ = optimize_acqf(acq_function=oneEI_f,
                                           bounds=bounds,
                                           q=1,
                                           num_restarts=num_restarts,
                                           raw_samples=raw_samples)
        sampler = IIDNormalSampler(num_samples=num_samples, resample=False, seed=seed_out)
        inner_sampler = IIDNormalSampler(num_samples=M, resample=False, seed=seed)
        oneEI_c = OneStepIncAntEI(model=model,
                                  sampler=sampler,
                                  inner_sampler=inner_sampler,
                                  fc=1,
                                  bounds=bounds)

        new_candidate_c, _ = optimize_acqf(acq_function=oneEI_c,
                                           bounds=bounds,
                                           q=1,
                                           num_restarts=num_restarts,
                                           raw_samples=raw_samples)
        new_candidate = new_candidate_f - new_candidate_c
    return new_candidate


# ---------------------------------------------------------------------------- #
def test_fn_2ei_ant_inc(train_x, train_y, l, dl, num_samples, num_restarts, raw_samples, bounds):
    r"""
    Antithetic increment 2-step EI

    Parameters
    ------
    train_x
    train_y
    l:          dl + l is the level
    dl:         starting level
    """
    M = 2 ** (l + dl)
    model = GPmll(train_x, train_y)
    base_samples1 = torch.randn(
        num_samples, device=train_x.device, dtype=train_x.dtype
    )
    base_samples2 = torch.randn(
        M, device=train_x.device, dtype=train_x.dtype
    )
    if l == 0:
        sampler1 = IIDNormalSampler(num_samples=num_samples, resample=False)
        sampler1.base_samples = base_samples1
        sampler2 = IIDNormalSampler(num_samples=M, resample=False)
        sampler2.base_samples = base_samples2

        twoEI = TwoStepIncAntEI(model=model,
                                samplers=[sampler1, sampler2],
                                bounds=bounds,
                                fc=0,)

        q = twoEI.get_augmented_q_batch_size(1)
        new_candidate, _ = optimize_acqf(acq_function=twoEI,
                                         bounds=bounds,
                                         q=q,
                                         num_restarts=num_restarts,
                                         raw_samples=raw_samples)

    else:
        sampler1 = IIDNormalSampler(num_samples=num_samples, resample=False)
        sampler1.base_samples = base_samples1
        sampler2f = IIDNormalSampler(num_samples=M, resample=False)
        sampler2f.base_samples = base_samples2
        sampler2c1 = IIDNormalSampler(num_samples=M//2, resample=False)
        sampler2c1.base_samples = base_samples2[1::2]
        sampler2c2 = IIDNormalSampler(num_samples=M//2, resample=False)
        sampler2c2.base_samples = base_samples2[::2]

        twoEI_f = TwoStepIncAntEI(model=model,
                                  samplers=[sampler1, sampler2f],
                                  bounds=bounds,
                                  fc=0)

        q = twoEI_f.get_augmented_q_batch_size(1)
        new_candidate_f, _ = optimize_acqf(acq_function=twoEI_f,
                                           bounds=bounds,
                                           q=q,
                                           num_restarts=num_restarts,
                                           raw_samples=raw_samples)

        twoEI_c = TwoStepIncAntEI(model=model,
                                  samplers=[sampler1, sampler2c1, sampler2c2],
                                  bounds=bounds,
                                  fc=1)

        q = twoEI_c.get_augmented_q_batch_size(1)
        new_candidate_c, _ = optimize_acqf(acq_function=twoEI_c,
                                           bounds=bounds,
                                           q=q,
                                           num_restarts=num_restarts,
                                           raw_samples=raw_samples)
        new_candidate = new_candidate_f - new_candidate_c
    return new_candidate
# ---------------------------------------------------------------------------- #
