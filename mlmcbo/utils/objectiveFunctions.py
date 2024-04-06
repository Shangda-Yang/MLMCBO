import math

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

class Goldstein(SyntheticTestFunction):

    dim = 2
    _bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    _optimal_value = 3.
    _optimizers = [(0., -1.)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = 1. + pow((x1+x2+1.), 2)*(19-14*x1+3*pow(x1, 2)-14*x2+6*x1*x2+3*pow(x2, 2))
        term2 = 30. + pow((2*x1-3*x2), 2)*(18.-32.*x1+12*pow(x1, 2)+48*x2-36.*x1*x2+27*pow(x2, 2))
        return term1 * term2


class Ackley5(SyntheticTestFunction):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 5,
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
        self.dim = dim
        if bounds is None:
            bounds = [(-1., 1.) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X: Tensor) -> Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        return part1 + part2 + a + math.e

class Levy10(SyntheticTestFunction):
    r"""Levy synthetic test function.

    d-dimensional function (usually evaluated on `[-10, 10]^d`):

        f(x) = sin^2(pi w_1) +
            sum_{i=1}^{d-1} (w_i-1)^2 (1 + 10 sin^2(pi w_i + 1)) +
            (w_d - 1)^2 (1 + sin^2(2 pi w_d))

    where `w_i = 1 + (x_i - 1) / 4` for all `i`.

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim=10,
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
        self.dim = dim
        if bounds is None:
            bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            dim=-1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
        )
        return part1 + part2 + part3

class SixHumpCamel2(SyntheticTestFunction):

    dim = 2
    _bounds = [(-2.0, 2.0), (-1.0, 1.0)]
    _optimal_value = 0.
    _optimizers = [(0.0898, -0.7126), (-0.0898, 0.7126)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return (
                ((4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
            + x1 * x2
            + (4 * x2**2 - 4) * x2**2 + 1.0316)/2
        )


class Cosine(SyntheticTestFunction):
    r"""Cosine Mixture test function.

    8-dimensional function (usually evaluated on `[-1, 1]^8`):

        f(x) = 0.1 sum_{i=1}^8 cos(5 pi x_i) - sum_{i=1}^8 x_i^2

    f has one maximizer for its global maximum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0.8`
    """

    dim = 8
    _bounds = [(-1.0, 1.0) for _ in range(8)]
    _optimal_value = -0.4
    _optimizers = [tuple(0.0 for _ in range(8))]

    def evaluate_true(self, X: Tensor) -> Tensor:
        results = torch.sum(0.1 * torch.cos(5 * math.pi * X) - X**2, dim=-1)/2.
        results += torch.sum(0.5*math.pi*torch.sin(5*math.pi*X) + 2*X, dim=-1)
        return results

class Branin2(SyntheticTestFunction):
    r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """

    dim = 2
    _bounds = [(0.0, 15.0), (-5.0, 15.0)]
    _optimal_value = 0.397887
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi**2) * X[..., 0] ** 2
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1**2 + t2 + 10


