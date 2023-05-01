import numpy as np
import torch
from typing import Optional, Tuple, Union, Any, List

from botorch import settings
from botorch.acquisition import ExpectedImprovement, qKnowledgeGradient, \
    qExpectedImprovement, MCAcquisitionFunction, OneShotAcquisitionFunction,\
    multi_step_lookahead
from botorch.models.model import Model
from botorch.sampling import MCSampler, IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points, match_batch_shape
from torch import Tensor, Size

from botorch.acquisition import qMultiStepLookahead
from botorch.acquisition.multi_step_lookahead import _construct_sample_weights


class TwoStepIncAntEI(qMultiStepLookahead):

    def __init__(
            self,
            model: Model,
            bounds: Tensor,
            samplers: List[MCSampler],
            fc: int,
            **kwargs
    ) -> None:
        self.k = 2
        batch_sizes = [1, 1]
        super().__init__(model=model,
                         batch_sizes=batch_sizes,
                         samplers=samplers,
                         **kwargs)
        if fc == 1:
            num_fantasies = [samplers[i].sample_shape[0] for i in range(len(samplers)-1)]
            self.num_fantasies = num_fantasies
        self.bounds = bounds
        self.fc = fc


    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xs = self.get_multi_step_tree_input_representation(X)

        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        X0 = Xs[0]
        sample_weights = torch.ones(*X0.shape[:-2], device=X.device, dtype=X.dtype)
        best_f0 = self.model.train_targets.max()
        stage_val_func0 = ExpectedImprovement(model=self.model, best_f=best_f0)
        stage_val0 = stage_val_func0(X=X0)
        running_val = stage_val0

        fantasy_model1 = self.model.fantasize(
            X=X0, sampler=self.samplers[0], observation_noise=False, propagate_grads=True
        )

        sample_weights = _construct_sample_weights(
            prev_weights=sample_weights, sampler=self.samplers[0]
        )
        if self.fc == 0:

            X1 = Xs[1]
            best_f1 = fantasy_model1.train_targets.max()
            stage_val_func1 = ExpectedImprovement(model=fantasy_model1, best_f=best_f1)
            stage_val1 = stage_val_func1(X=X1)
            running_val = running_val + stage_val1

            X2 = Xs[2]
            fantasy_model2 = fantasy_model1.fantasize(
                X=X1, sampler=self.samplers[1], observation_noise=False, propagate_grads=True
            )
            sample_weights = _construct_sample_weights(
                prev_weights=sample_weights, sampler=self.samplers[1]
            )

            best_f2 = fantasy_model2.train_targets.max()
            stage_val_func2 = ExpectedImprovement(model=fantasy_model2, best_f=best_f2)
            stage_val2 = stage_val_func2(X=X2)
            running_val = running_val + stage_val2

            batch_shape = running_val.shape[self.k:]
            sample_weights = sample_weights.expand(running_val.shape)
            return (running_val * sample_weights).view(-1, *batch_shape).sum(dim=0)
        else:
            X11 = Xs[1]
            best_f1 = fantasy_model1.train_targets.max()
            stage_val_func11 = ExpectedImprovement(model=fantasy_model1, best_f=best_f1)
            stage_val11 = stage_val_func11(X=X11)
            running_val1 = running_val + stage_val11

            X21 = Xs[2]
            fantasy_model21 = fantasy_model1.fantasize(
                X=X11, sampler=self.samplers[1], observation_noise=False, propagate_grads=True
            )

            best_f21 = fantasy_model21.train_targets.max()
            stage_val_func21 = ExpectedImprovement(model=fantasy_model21, best_f=best_f21)
            stage_val21 = stage_val_func21(X=X21)
            running_val1 = running_val1 + stage_val21

            batch_shape1 = running_val1.shape[self.k:]
            sample_weights1 = sample_weights.expand(running_val1.shape)

            X12 = Xs[3]
            stage_val_func12 = ExpectedImprovement(model=fantasy_model1, best_f=best_f1)
            stage_val12 = stage_val_func12(X=X12)
            running_val2 = running_val + stage_val12

            X22 = Xs[4]
            fantasy_model22 = fantasy_model1.fantasize(
                X=X12, sampler=self.samplers[2], observation_noise=False, propagate_grads=True
            )

            best_f22 = fantasy_model22.train_targets.max()
            stage_val_func22 = ExpectedImprovement(model=fantasy_model22, best_f=best_f22)
            stage_val22 = stage_val_func22(X=X22)
            running_val2 = running_val2 + stage_val22

            batch_shape2 = running_val2.shape[self.k:]
            sample_weights2 = sample_weights.expand(running_val2.shape)

            return ( (running_val1 * sample_weights1).view(-1, *batch_shape1).sum(dim=0) + \
                    (running_val2 * sample_weights2).view(-1, *batch_shape2).sum(dim=0) ) / 2


    @property
    def _num_auxiliary(self) -> int:
        r"""Number of auxiliary variables in the q-batch dimension.

        Returns:
             `q_aux` s.t. `q + q_aux = augmented_q_batch_size`
        """
        num_auxiliary = np.dot(self.batch_sizes, np.cumprod(self.num_fantasies)).item()

        if self.fc == 1:
            return 2*num_auxiliary
        else:
            return num_auxiliary

    def get_multi_step_tree_input_representation(self, X: Tensor) -> List[Tensor]:
        r"""Get the multi-step tree representation of X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.

        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        if self.fc == 1:
            sizes += sizes[1:]
            shapes += shapes[1:]
        Xsplit = torch.split(X, sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)

        return [X0] + Xother