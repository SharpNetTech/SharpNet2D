#!/usr/bin/env false
# -*- coding: utf-8 -*-

""" SharpNet implementation """

import torch
import torch.nn as nn
from typing import Optional
from .mlp import MLP
from .bem import BEMquery
from .encoding import Encoder

class SharpNet(nn.Module):
    def __init__(
        self,
        mlp: MLP,
        bem: Optional[BEMquery] = None,
        input_enc: Optional[Encoder] = None,
        feat_enc: Optional[Encoder] = None,
    ):
        super().__init__()
        self.mlp = mlp
        self.bem = bem
        self.input_enc = input_enc
        self.feat_enc = feat_enc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # normalize the input shape
        orig_shape = inputs.shape
        inputs = inputs.reshape(-1, inputs.shape[-1])

        # run BEMquery, if available
        features = None
        if self.bem is not None:
            features = self.bem.forward(inputs)

        # Run encodings, if available
        if self.input_enc is not None:
            inputs = self.input_enc.forward(inputs)
        if self.feat_enc is not None and features is not None:
            features = self.feat_enc.forward(features)

        outputs = self.mlp.forward(inputs, features)
        # restore original shape
        outputs = outputs.reshape([*orig_shape[:-1], outputs.shape[-1]])
        return outputs

    def gradient(self, pt: torch.Tensor) -> torch.Tensor:
        pt.requires_grad_(True)
        y = self.forward(pt)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=pt,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients
