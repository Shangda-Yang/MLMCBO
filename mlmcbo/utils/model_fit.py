import torch
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel


def GPmodel(train_x, train_y, bounds=None, mod='Matern'):
    # print(train_y)
    if mod == 'Matern':
        train_yvar = torch.full_like(train_y, 1e-4)
        model = FixedNoiseGP(train_x,
                             train_y,
                             train_yvar,
                             # input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
                             # outcome_transform=Standardize(m=1)
                             )
    else:
        train_yvar = torch.full_like(train_y, 1e-3)
        kernel = ScaleKernel(RBFKernel())
        # outcome_transform = Standardize(m=1)
        model = FixedNoiseGP(train_x,
                             train_y,
                             train_yvar,
                             covar_module=kernel,
                             input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
                             outcome_transform=Standardize(m=1)
                             )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

