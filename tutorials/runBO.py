import time

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.acquisition.multi_step_lookahead import make_best_f, qMultiStepLookahead
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler

from mlmcbo.acquisition_functions import qEIMLMCOneStep, qEIMLMCTwoStep
from mlmcbo.acquisition_functions import CustomqMultiStepLookahead
from mlmcbo.utils import optimize_mlmc
from mlmcbo.utils.model_fit import GPmodel
from mlmcbo.utils.optimize_mlmc import optimize_mlmc_two


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
                 q=None,
                 ML=True,
                 dl=0,
                 match_mode='point'):
        if q is None:
            q = [1, 2]
        self.target = target
        self.train_x = train_x
        self.train_y = train_y
        self.n_runs = n_runs
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.eps = eps
        self.q1 = q[0]
        self.q2 = q[1]
        self.dl = dl
        self.ML = ML
        self.match_mode = match_mode

    def run(self):
        results = torch.zeros(1, self.n_runs)
        costs = torch.zeros(1, self.n_runs)

        for j in range(self.n_runs):
            # fit Gaussian process
            model = GPmodel(self.train_x, self.train_y)
            if self.ML:
                # MLMC BO
                qEI = qEIMLMCOneStep(
                    model=model,
                    bounds=self.bounds,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    q=self.q1,
                    batch_sizes=[self.q2]
                )
                start_time = time.time()
                new_candidate, _, _ = optimize_mlmc(inc_function=qEI,
                                                    eps=self.eps,
                                                    dl=self.dl,
                                                    alpha=1,
                                                    beta=1.5,
                                                    gamma=1,
                                                    meanc=1,
                                                    varc=1,
                                                    var0=1,
                                                    match_mode=self.match_mode,
                                                    )

                costs[:, j] = time.time() - start_time
                print("Iteration {} finished, MLMC time {:.4f}".format(j, costs[:, j].item()))
            else:
                # MC BO
                num_samples = np.round((1 / np.power(self.eps, 2))).astype(int)
                sampler = IIDNormalSampler(sample_shape=torch.Size([num_samples]))
                samplers = [sampler]
                if self.q1 == 1:
                    inner_mc_samplers = [None, IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
                    valfunc_cls = [ExpectedImprovement, qExpectedImprovement]
                else:
                    inner_mc_samplers = [sampler, IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
                    valfunc_cls = [qExpectedImprovement, qExpectedImprovement]
                valfunc_argfacs = [make_best_f, make_best_f]
                qEI = CustomqMultiStepLookahead(model=model,
                                                batch_sizes=[self.q2],
                                                samplers=samplers,
                                                inner_mc_samplers=inner_mc_samplers,
                                                valfunc_cls=valfunc_cls,
                                                valfunc_argfacs=valfunc_argfacs)
                q = qEI.get_augmented_q_batch_size(self.q1)
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

            # model = GPmodel(self.train_x, self.train_y)
            # qEI.model = model

            best_candidate = self.train_x[torch.argmax(self.train_y)]
            best_result = self.target(best_candidate).unsqueeze(-1)

            results[:, j] = best_result
        return results, costs


class runBOTwo():
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
                 q=None,
                 ML=True,
                 dl=0,
                 match_mode='point'):
        if q is None:
            q = [1, 1, 1]
        self.target = target
        self.train_x = train_x
        self.train_y = train_y
        self.n_runs = n_runs
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.eps = eps
        self.q1 = q[0]
        self.q2 = q[1:]
        self.dl = dl
        self.ML = ML
        self.match_mode = match_mode

    def run(self):
        results = torch.zeros(1, self.n_runs)
        costs = torch.zeros(1, self.n_runs)

        for j in range(self.n_runs):
            # fit Gaussian process
            model = GPmodel(self.train_x, self.train_y)
            if self.ML:
                # MLMC BO
                qEI = qEIMLMCTwoStep(
                    model=model,
                    bounds=self.bounds,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    q=self.q1,
                    batch_sizes=self.q2
                )
                start_time = time.time()
                new_candidate, _, _ = optimize_mlmc_two(inc_function=qEI,
                                                        eps=self.eps,
                                                        dl=self.dl,
                                                        alpha=1,
                                                        beta=1.5,
                                                        gamma=1,
                                                        meanc=1,
                                                        varc=1,
                                                        var0=1,
                                                        match_mode=self.match_mode,
                                                        )

                costs[:, j] = time.time() - start_time
                print("Iteration {} finished, MLMC time {:.4f}".format(j, costs[:, j].item()))
            else:
                # MC BO
                num_samples = np.round((1 / np.power(self.eps, 2))).astype(int)
                # sampler = IIDNormalSampler(sample_shape=torch.Size([num_samples]))
                samplers = [IIDNormalSampler(sample_shape=torch.Size([num_samples])),
                            IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
                # if self.q1 == 1:
                #     inner_mc_samplers = [None, IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
                #     valfunc_cls = [ExpectedImprovement, qExpectedImprovement]
                # else:
                #     inner_mc_samplers = [sampler, IIDNormalSampler(sample_shape=torch.Size([num_samples]))]
                #     valfunc_cls = [qExpectedImprovement, qExpectedImprovement]
                valfunc_cls = [ExpectedImprovement, ExpectedImprovement, ExpectedImprovement]
                valfunc_argfacs = [make_best_f, make_best_f, make_best_f]
                # qEI = CustomqMultiStepLookahead(model=model,
                #                                 batch_sizes=[self.q2],
                #                                 samplers=samplers,
                #                                 inner_mc_samplers=inner_mc_samplers,
                #                                 valfunc_cls=valfunc_cls,
                #                                 valfunc_argfacs=valfunc_argfacs)
                qEI = qMultiStepLookahead(model=model,
                                          batch_sizes=self.q2,
                                          samplers=samplers,
                                          valfunc_cls=valfunc_cls,
                                          valfunc_argfacs=valfunc_argfacs)
                q = qEI.get_augmented_q_batch_size(self.q1)
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

            # model = GPmodel(self.train_x, self.train_y)
            # qEI.model = model

            best_candidate = self.train_x[torch.argmax(self.train_y)]
            best_result = self.target(best_candidate).unsqueeze(-1)

            results[:, j] = best_result
        return results, costs