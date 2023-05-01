import torch
from botorch import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, RBFKernel
from botorch import fit_gpytorch_model
from botorch.acquisition.multi_step_lookahead import make_best_f, qMultiStepLookahead
from botorch.optim.optimize import optimize_acqf
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.sampling import IIDNormalSampler
from botorch.acquisition.analytic import ExpectedImprovement
from GPmodels import GPmll

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.double

# ----------------------------------------------------------------------- #
def get_next_point_twoEI(train_x, train_y, bounds, num_samples,
                         num_restarts=20, raw_samples=512):
    r"""One step lookahead EI
    """
    model = GPmll(train_x, train_y)
    q = len(num_samples) + 1
    batch_sizes = [1] * (q - 1)
    samplers = [IIDNormalSampler(num_samples=n, resample=False) for n in num_samples]
    valfunc_cls = [ExpectedImprovement for _ in range(len(num_samples)+1)]
    valfunc_argfacs = [make_best_f for _ in range(len(num_samples)+1)]
    twoEI = qMultiStepLookahead(model=model,
                                batch_sizes=batch_sizes,
                                num_fantasies=None,
                                samplers=samplers,
                                valfunc_cls=valfunc_cls,
                                valfunc_argfacs=valfunc_argfacs,
                                inner_mc_samples=None)
    q_prime = twoEI.get_augmented_q_batch_size(1)
    new_candidate, new_value = optimize_acqf(acq_function=twoEI,
                                             bounds=bounds,
                                             q=q_prime,
                                             num_restarts=num_restarts,
                                             raw_samples=raw_samples)
    return new_candidate, new_value
# ----------------------------------------------------------------------- #
