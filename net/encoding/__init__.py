# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

class Encoder(nn.Module, metaclass=ABCMeta):
    """ Base class for all encoders. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

from .pe import PositionalEncoder
from .ingp import INGPEncoder
