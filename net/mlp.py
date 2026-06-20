#!/usr/bin/env false
# -*- coding: utf-8 -*-

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

class MLP(nn.Module):
    def __init__(
        self, *,
        d_in: int,                      # width of input
        d_feat: int,                    # width of feature
        d_out: int,                     # width of output
        d_hidden: int,                  # width of hidden layer
        n_layers: int,                  # number of hidden layers
        nonlinearity: str = 'relu',     # type of activation
        weight_norm: bool = True,       # weight normalization
        shift: float = 0.0,             # output shift
        **kwargs
    ):
        super(MLP, self).__init__()
        if len(kwargs.keys()) > 0:
            key_list = list(map(lambda x : '`' + str(x) + '`', list(kwargs.keys())))
            logging.warning("MLP received extra arguments: {}".format(', '.join(key_list)))

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        # set up proper feature insertion
        if d_feat > 0:
            dims[0] += d_feat

        # some variable registration
        self.num_layers = len(dims)     # n_layers + input + output = n_layers + 2
        self.shift = shift
        self.d_feat = d_feat

        # set up activation
        if nonlinearity.casefold() == 'relu':
            logging.info("Info: Nonlinearity set to ReLU.")
            self.activation = nn.ReLU()
        elif nonlinearity.casefold() == 'softplus':
            logging.info("Info: Nonlinearity set to Softplus(beta=100.0)")
            self.activation = nn.Softplus(beta=100.0)
        else:
            raise ValueError(f"Unrecognised activaty function: `{nonlinearity}`.")

        for l in range(0, self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

    def forward(self, inputs: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # set up initial input tensor
        x = inputs
        if features is not None:
            x = torch.cat([inputs, features], dim=1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x + self.shift
