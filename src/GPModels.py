from botorch.utils import torch
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel


# model definition
def GPmll(train_x, train_y, model=None, kernel=None):
    train_yvar = torch.full_like(train_y, 1e-4)
    if kernel == "RBF":
        covar_module = ScaleKernel(base_kernel=RBFKernel())
        if model == "RandomNoise":
            model = SingleTaskGP(train_x, train_y, covar_module=covar_module)
        else:
            model = SingleTaskGP(train_x, train_y, train_yvar, covar_module=covar_module)

    else:
        if model == "RandomNoise":
            model = SingleTaskGP(train_x, train_y)
        else:
            model = FixedNoiseGP(train_x, train_y, train_yvar)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

