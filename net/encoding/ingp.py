# -*- encoding: utf-8 -*-

import logging
import torch
from typing import List
from . import Encoder

try:
    import tinycudann as tcnn

    class INGPEncoder(Encoder):
        """ Wrapper around tinycudann Encoding module. Add support for input scaling and shifting. """

        _inner: tcnn.Encoding

        def __init__(
            self, *,
            scale: float = 1.0,
            shift: List[float] = [0.0, 0.0],
            **kwargs,
        ):
            super().__init__()
            self._scale = scale
            self._shift = torch.tensor(shift, dtype=torch.float32).view(1, -1)
            self._inner = tcnn.Encoding(**kwargs)

        def output_dim(self) -> int:
            return self._inner.n_output_dims

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x * self._scale + self._shift.to(x.device)
            return self._inner(x)

except ImportError:
    logging.warning("tinycudann is not installed. INGPEncoder will not be available if you ever call it.")

    class INGPEncoder(Encoder):
        """ Stub class for when tinycudann is not installed. Its only purpose is to raise an error when instantiated. """

        def __init__(self, **kwargs):
            super().__init__()
            raise ImportError("tinycudann is not installed. Please install tinycudann to use INGPEncoder.")

        def output_dim(self) -> int:
            raise NotImplementedError("tinycudann is not installed. Please install tinycudann to use INGPEncoder.")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("tinycudann is not installed. Please install tinycudann to use INGPEncoder.")
