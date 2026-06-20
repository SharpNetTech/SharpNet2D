#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os, sys
    SCRIPTNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPTNAME)
    SCRIPTNAME = os.path.dirname(SCRIPTNAME)
    sys.path.append(SCRIPTNAME)
    SCRIPTNAME = os.path.dirname(SCRIPTNAME)
    sys.path.append(SCRIPTNAME)

import logging
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import Line, Path
import torch
from typing import Optional, Tuple

from data import Dataset
from data import SharpFeature as MedialAxis

class MedialRectangle(Dataset):
    def __init__(self, *, x1: float, y1: float, x2: float, y2: float, **kwargs):
        super(MedialRectangle, self).__init__()
        if kwargs.keys().__len__() > 0:
            logging.warning(f"MedialRectangle received extra arguments: {kwargs.keys()}")

        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

        width = self.x2 - self.x1
        height = self.y2 - self.y1
        diff = abs(width - height) / 2.0
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2

        if width > height:
            self.medialaxis = MedialAxis(Path(
                Line(complex(self.x1, self.y1), complex(cx - diff, cy)),
                Line(complex(self.x2, self.y1), complex(cx + diff, cy)),
                Line(complex(self.x1, self.y2), complex(cx - diff, cy)),
                Line(complex(self.x2, self.y2), complex(cx + diff, cy)),
                Line(complex(cx - diff, cy), complex(cx + diff, cy)),
            ))
        elif width < height:
            self.medialaxis = MedialAxis(Path(
                Line(complex(self.x1, self.y1), complex(cx, cy - diff)),
                Line(complex(self.x2, self.y1), complex(cx, cy - diff)),
                Line(complex(self.x1, self.y2), complex(cx, cy + diff)),
                Line(complex(self.x2, self.y2), complex(cx, cy + diff)),
                Line(complex(cx, cy - diff), complex(cx, cy + diff)),
            ))
        else:
            self.medialaxis = MedialAxis(Path(
                Line(complex(self.x1, self.y1), complex(cx, cy)),
                Line(complex(self.x2, self.y1), complex(cx, cy)),
                Line(complex(self.x1, self.y2), complex(cx, cy)),
                Line(complex(self.x2, self.y2), complex(cx, cy)),
            ))

    def dist_1(self, x: torch.Tensor) -> torch.Tensor:
        """ Area 1: the point is inside the rectange """
        candidates = torch.stack([
            x[:, 0] - self.x1, self.x2 - x[:, 0],
            x[:, 1] - self.y1, self.y2 - x[:, 1]
        ], dim=1)
        return torch.min(candidates, dim=1, keepdim=True)[0]

    def dist_2(self, x: torch.Tensor) -> torch.Tensor:
        """ Area 2: the point is outside the rectangle, but projects to the horizontal edges of the rectangle """
        candidates = torch.stack([
            torch.abs(x[:, 1] - self.y1), torch.abs(x[:, 1] - self.y2)
        ], dim=1)
        return torch.min(candidates, dim=1, keepdim=True)[0]

    def dist_3(self, x: torch.Tensor) -> torch.Tensor:
        """ Area 3: the point is outside the rectangle, but projects to the vertical edges of the rectangle """
        candidates = torch.stack([
            torch.abs(x[:, 0] - self.x1), torch.abs(x[:, 0] - self.x2)
        ], dim=1)
        return torch.min(candidates, dim=1, keepdim=True)[0]

    def dist_4(self, x: torch.Tensor) -> torch.Tensor:
        """ Area 4: the point is outside the rectangle, and projects to the corners of the rectangle """
        candidates = torch.stack([
            (x[:, 0] - self.x1) ** 2 + (x[:, 1] - self.y1) ** 2,
            (x[:, 0] - self.x1) ** 2 + (x[:, 1] - self.y2) ** 2,
            (x[:, 0] - self.x2) ** 2 + (x[:, 1] - self.y1) ** 2,
            (x[:, 0] - self.x2) ** 2 + (x[:, 1] - self.y2) ** 2
        ], dim=1)
        return torch.sqrt(torch.min(candidates, dim=1, keepdim=True)[0])

    @torch.no_grad()
    def forward(self, x: torch.Tensor, keepdim: bool=False) -> torch.Tensor:
        """ Get the distance to the edges of the rectangle """
        assert x.shape[-1] == 2, "Input must have 2 elements in the last dimension"
        shape = x.shape[:-1]
        x = x.reshape(-1, 2)
        numpts = x.shape[0]

        # Determine by region
        area_1 = (x[:, 0] >= self.x1) & (x[:, 0] <= self.x2) & (x[:, 1] >= self.y1) & (x[:, 1] <= self.y2)
        area_2 = (~area_1) & (x[:, 0] >= self.x1) & (x[:, 0] <= self.x2)
        area_3 = (~area_1) & (x[:, 1] >= self.y1) & (x[:, 1] <= self.y2)
        area_4 = ~(area_1 | area_2 | area_3)

        area_1 = area_1.unsqueeze(-1)
        area_2 = area_2.unsqueeze(-1)
        area_3 = area_3.unsqueeze(-1)
        area_4 = area_4.unsqueeze(-1)

        # inside = positive distance
        # outside = negative distance
        out = torch.where(area_1,  self.dist_1(x), torch.zeros_like(x[:, :1])) + \
              torch.where(area_2, -self.dist_2(x), torch.zeros_like(x[:, :1])) + \
              torch.where(area_3, -self.dist_3(x), torch.zeros_like(x[:, :1])) + \
              torch.where(area_4, -self.dist_4(x), torch.zeros_like(x[:, :1]))

        out = torch.reshape(out, (*shape, 1))
        if not keepdim:
            out = out.squeeze(-1)
        return out

    def inside_polygon(self, x: torch.Tensor) -> torch.Tensor:
        """ Check if the points are inside the rectangle """

        inside = (x[:, 0] >= self.x1) & (x[:, 0] <= self.x2) & (x[:, 1] >= self.y1) & (x[:, 1] <= self.y2)
        return inside

    def approx_ridge(self, *, bbox=None, segment_length=-1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v, e, n = self.medialaxis.as_mesh_by_length(segment_length)
        v = torch.from_numpy(v)
        e = torch.from_numpy(e)
        n = torch.from_numpy(n)
        return v, e, n

    def generate_figure(self, canvas: Optional[Axes], *, resolution, bbox, vmin=None, vmax=None, **kwargs) -> np.ndarray:
        """
        bbox, roi: (x min, x max, y min, y max)
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
