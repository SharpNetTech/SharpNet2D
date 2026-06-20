# -*- coding: utf-8 -*-

from collections.abc import Callable
import logging
import torch
from typing import Optional, List
from . import Encoder

class PositionalEncoder(Encoder):

    _embed_fns: List[Callable[[torch.Tensor], torch.Tensor]]
    _out_dim: int

    def __init__(
        self, *,
        include_input: bool = True,
        input_dims: int,
        max_freq_log2: Optional[int] = None,
        num_freqs: int,
        log_sampling: bool = True,
        periodic_fns: List[Callable[[torch.Tensor], torch.Tensor]] = [torch.sin, torch.cos],
        **kwargs,
    ):
        super().__init__()
        if kwargs.keys().__len__() > 0:
            logging.warning(f"PositionalEncoder received extra arguments: {kwargs.keys()}")

        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = max_freq_log2 if max_freq_log2 is not None else num_freqs - 1
        logging.info(f"Setting max_freq_log2 = {max_freq}")
        N_freqs = num_freqs

        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self._embed_fns = embed_fns
        self._out_dim = out_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(inputs) for fn in self._embed_fns], -1)

    def output_dim(self) -> int:
        return self._out_dim
