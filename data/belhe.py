#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os, sys
    SCRIPTNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPTNAME)
    sys.path.append(os.path.dirname(SCRIPTNAME))

import logging
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import Path, Line
import torch
import torch.nn as nn
from typing import Optional, Tuple
from data import Dataset, SharpFeature

class Belhe(Dataset):
    def __init__(self, **kwargs):
        super(Belhe, self).__init__()
        if kwargs.keys().__len__() > 0:
            logging.warning(f"Belhe received extra arguments: {kwargs.keys()}")

        verts = np.array([
            # For the line
            [0.27721, 0.42631],
            [0.40487, 0.79068],
            # For the polygon
            [0.45444, 0.42631],
            [0.83864, 0.39781],
            [0.91548, 0.81547],
            [0.40487, 0.57752],
        ], dtype=np.float32)
        edges = np.array([
            [0, 1],
            [2, 3], [3, 4], [4, 5], [5, 2],
        ], dtype=np.int32)

        self._vertices = nn.Parameter(torch.from_numpy(verts), requires_grad=False)
        self._edges = nn.Parameter(torch.from_numpy(edges), requires_grad=False)

        self._sharp_features = SharpFeature(Path(
            Line(complex(verts[0, 0], verts[0, 1]), complex(verts[1, 0], verts[1, 1])),
            Line(complex(verts[2, 0], verts[2, 1]), complex(verts[3, 0], verts[3, 1])),
            Line(complex(verts[3, 0], verts[3, 1]), complex(verts[4, 0], verts[4, 1])),
            Line(complex(verts[4, 0], verts[4, 1]), complex(verts[5, 0], verts[5, 1])),
            Line(complex(verts[5, 0], verts[5, 1]), complex(verts[2, 0], verts[2, 1])),
        ))

    @staticmethod
    def _C0_2d(pt: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        def _dot(a: torch.Tensor, b: torch.Tensor)  ->torch.Tensor:
            return torch.sum(a*b, dim=-1, keepdim=True)
        def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a0 = a[:,:,[0]]
            a1 = a[:,:,[1]]
            b0 = b[:,:,[0]]
            b1 = b[:,:,[1]]
            return a0 * b1 - a1 * b0

        v = face-pt.unsqueeze(dim=-2) # P x F x 2 x 2
        va = v[:,:,0,:] # P x F x 2
        vb = v[:,:,1,:]
        vd = face[:,:,1,:] - face[:,:,0,:]
        ld = vd.norm(dim=-1, keepdim=True) # P x F x 1
        l = torch.abs(_cross2d(va, vb)) / ld
        L_ = torch.square(l)
        l_ = torch.sqrt(L_)
        def f(t):
            return t * (torch.log(torch.square(t) + L_) - 2) / 2 + l_ * torch.atan2(t, l_)
        res = f(_dot(vb,vd) / ld) - f(_dot(va,vd) / ld)
        return res.squeeze(dim=-1) / torch.pi # P x F

    def forward(self, x: torch.Tensor, keepdim: bool=False) -> torch.Tensor:
        assert x.shape[-1] == 2, "Input must have 2 elements in the last dimension"
        shape = x.shape[:-1]
        x = x.reshape(-1, 2)
        rpt = x.unsqueeze(dim=1) # P x {1} x 2

        vertices = self._vertices.detach()
        edges = self._edges.detach()
        # A lite version of BEMquery that solves the laplacian equation.
        # The data is small enough for dense non-mollifier implementation.

        redge = torch.index_select(vertices, 0, edges.flatten()).reshape(5, 2, 2).unsqueeze(dim=0)  # (1, num_edges, 2, 2)
        out = self._C0_2d(rpt, redge).sum(dim=-1, keepdim=True)   # P, 1

        out = torch.reshape(out, (*shape, -1))
        if not keepdim:
            out = out.squeeze(-1)
        return out

    def approx_ridge(self, segment_length=-1.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v, e, n = self._sharp_features.as_mesh_by_length(segment_length)
        v = torch.from_numpy(v).detach().to(torch.float32)
        e = torch.from_numpy(e).detach().to(torch.int32)
        n = torch.from_numpy(n).detach().to(torch.int32)
        return v, e, n

    def generate_figure(self, canvas: Optional[Axes], *, resolution, bbox, vmin=None, vmax=None, **kwargs) -> np.ndarray:
        """
        canvas, roi: (x min, x max, y min, y max)
        """

        c_xmin, c_xmax, c_ymin, c_ymax = bbox
        query_func = torch.func.vmap(torch.func.vmap(lambda x: self.forward(x, keepdim=False)))

        x = torch.linspace(c_xmin, c_xmax, resolution)
        y = torch.linspace(c_ymin, c_ymax, resolution)
        xs, ys = torch.meshgrid(x, y, indexing="xy")
        pts = torch.stack([xs, ys], dim=-1)
        vals = query_func(pts).cpu().numpy()

        if canvas is not None:
            # Set canvas
            canvas.set_aspect("equal", adjustable="box")
            canvas.set_xlim(c_xmin, c_xmax)
            canvas.set_ylim(c_ymin, c_ymax)
            canvas.set_axis_off()

            # Draw color
            canvas.imshow(vals, cmap='winter', extent=bbox, origin='lower', vmin=vmin, vmax=vmax, zorder=0)

        return vals
