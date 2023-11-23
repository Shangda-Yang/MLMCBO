import torch
from torch import Tensor
from typing import List, Optional, Tuple
from botorch.test_functions import SyntheticTestFunction


class SelfDefinedFunction(SyntheticTestFunction):

    def __init__(
            self,
            noise_std: Optional[float] = None,
            negate: bool = False,
            bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-10.0, 10.0)]
        self._optimal_value = 1.4019
        self._optimizers = [(2.0087)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.exp(-(X - 2) ** 2) + torch.exp(-(X - 6) ** 2 / 10) + 1 / (X ** 2 + 1)
        return f.squeeze(-1)