#!/usr/bin/env false
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import svgpathtools
import torch
import torch.nn as nn
from typing import Tuple, List

class Dataset(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(Dataset, self).__init__()

    def output_dim(self) -> int:
        return 1

    @abstractmethod
    def forward(self, inputs: torch.Tensor, keepdim: bool = False, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def approx_ridge(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def gradient(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs.requires_grad_(True)
        y = self.forward(inputs)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=inputs,
            grad_outputs=d_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class SharpFeature():
    _paths: svgpathtools.Path

    def __init__(self, paths: svgpathtools.Path):
        self._paths = paths

    def scaled(self, scale: float) -> 'SharpFeature':
        new_path = copy.deepcopy(self._paths)
        new_path = new_path.scaled(scale)
        return SharpFeature(new_path)

    def translated(self, translation: np.ndarray) -> 'SharpFeature':
        new_path = copy.deepcopy(self._paths)
        new_path = new_path.translated(complex(translation[0], translation[1]))
        return SharpFeature(new_path)

    def as_mesh_by_length(self, length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return as a tuple of vertices, edges and nonmanifold point indices. Subsect each path into lengths """
        if length <= 0.0:
            return self.as_mesh_by_segments([1]*len(self._paths))
        segments = []
        for path in self._paths:
            pathlen = path.length()
            seg = pathlen / length
            segcand1 = max(1, np.floor(seg))    # Prevent div by zero
            segcand2 = np.ceil(seg)
            segcand1len = pathlen / segcand1
            segcand2len = pathlen / segcand2
            if (np.abs(segcand1len - length) <= np.abs(segcand2len - length)):
                segments.append(int(segcand1))
            else:
                segments.append(int(segcand2))

        return self.as_mesh_by_segments(segments)

    def as_mesh_by_segments(self, segments: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return as a tuple of vertices, edges and nonmanifold point indices. Subsect each path into given segments. """
        assert len(segments) == len(self._paths)
        points = []
        points_freq = {}
        edges = []
        # Find all endpoints for all paths
        path_endpoint_idx = []
        for path in self._paths:
            p0 = path.start
            p1 = path.end
            if p0 in points:
                idx0 = points.index(p0)
                points_freq[idx0] += 1
            else:
                idx0 = len(points)
                points.append(p0)
                points_freq[idx0] = 1
            if p1 in points:
                idx1 = points.index(p1)
                points_freq[idx1] += 1
            else:
                idx1 = len(points)
                points.append(p1)
                points_freq[idx1] = 1
            path_endpoint_idx.append([idx0, idx1])
 
        # Interpolate segments
        for path, segm, epidx in zip(self._paths, segments, path_endpoint_idx):
            if segm > 1:
                waypoint = np.linspace(0.0, 1.0, segm + 1)[1:-1]
                waypoint = [path.point(w) for w in waypoint]
                edges.append([epidx[0], len(points)])
                points.append(waypoint[0])
                if len(waypoint) > 1:
                    for w in waypoint[1:]:
                        edges.append([len(points)-1, len(points)])
                        points.append(w)
                edges.append([len(points)-1, epidx[1]])
            else:
                edges.append(epidx)

        # Convert into numpy arrays
        points = [[p.real, p.imag] for p in points]
        points = np.array(points, dtype=np.float32)
        edges = np.array(edges, dtype=np.int32)
        non_manifolds = [i for i, n in points_freq.items() if n != 2]
        non_manifolds = np.array(non_manifolds, dtype=np.int32)

        return points, edges, non_manifolds
