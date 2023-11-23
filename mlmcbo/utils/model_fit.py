import torch
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from gpytorch import ExactMarginalLogLikelihood


def GPmodel(train_x, train_y):
    train_yvar = torch.full_like(train_y, 1e-4)
    model = FixedNoiseGP(train_x, train_y, train_yvar)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

