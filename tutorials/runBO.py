import time

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.acquisition.multi_step_lookahead import make_best_f
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler

from mlmcbo.acquisition_functions import qEIMLMCOneStep
from mlmcbo.acquisition_functions.mlmc_inc_functions import CustomqMultiStepLookahead
from mlmcbo.utils import optimize_mlmc
from mlmcbo.utils.model_fit import GPmodel


class runBO():
    r"""run BO with MLMC or MC and return the results and costs"""
    def __init__(self,
                 target,
                 train_x,
                 train_y,
                 n_runs,
                 bounds,
                 num_restarts,
                 raw_samples,
                 eps,
                 ML=True,
                 dl=0):
        self.target = target
        self.train_x = train_x
        self.train_y = train_y
        self.n_runs = n_runs
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.eps = eps
        self.dl = dl
        self.ML = ML

    def run(self):
        results = torch.zeros(1, self.n_runs)
        costs = torch.zeros(1, self.n_runs)

        model = GPmodel(self.train_x, self.train_y)

        if self.ML:
            qEI = qEIMLMCOneStep(
                model=model,
                bounds=self.bounds,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples
            )
        else:
            num_samples = np.round((1 / np.power(self.eps, 2))).astype(int)
            samplers = [IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
            inner_mc_samplers = [None, IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
            valfunc_cls = [ExpectedImprovement, qExpectedImprovement]
            valfunc_argfacs = [make_best_f, make_best_f]
            qEI = CustomqMultiStepLookahead(model=model,
                                            batch_sizes=[2],
                                            samplers=samplers,
                                            inner_mc_samplers=inner_mc_samplers,
                                            valfunc_cls=valfunc_cls,
                                            valfunc_argfacs=valfunc_argfacs)
            q = qEI.get_augmented_q_batch_size(1)

        for j in range(self.n_runs):
            if self.ML:
                start_time = time.time()
                new_candidate, _, _ = optimize_mlmc(inc_function=qEI,
                                                    eps=self.eps,
                                                    dl=self.dl,
                                                    alpha=1,
                                                    beta=1.5,
                                                    gamma=1,
                                                    meanc=1,
                                                    varc=1,
                                                    var0=1
                                                    )

                costs[:, j] = time.time() - start_time
                print("Iteration {} finished, MLMC time {:.4f}".format(j, costs[:, j].item()))
            else:
                start = time.time()
                new_candidate, _ = optimize_acqf(acq_function=qEI,
                                                 bounds=self.bounds,
                                                 q=q,
                                                 num_restarts=self.num_restarts,
                                                 raw_samples=self.raw_samples,
                                                 )
                costs[:, j] = time.time() - start
                print("Iteration {} finished, MC time {:.4f}".format(j, costs[:, j].item()))

            new_result = self.target(new_candidate).unsqueeze(-1)
            self.train_x = torch.cat([self.train_x, new_candidate])
            self.train_y = torch.cat([self.train_y, new_result])

            model = GPmodel(self.train_x, self.train_y)
            qEI.model = model

            best_candidate = self.train_x[torch.argmax(self.train_y)]
            best_result = self.target(best_candidate).unsqueeze(-1)

            results[:, j] = best_result
        return results, costs

