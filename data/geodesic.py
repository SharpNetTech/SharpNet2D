#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os, sys
    SCRIPTNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPTNAME)
    sys.path.append(os.path.dirname(SCRIPTNAME))

import logging
import math
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple
from data import Dataset

class GeodesicCircle(Dataset):
    def __init__(self, *, cx: float=1.25, radius: float=1.0, **kwargs):
        super(GeodesicCircle, self).__init__()
        if kwargs.keys().__len__() > 0:
            logging.warning(f"GeodesicCircle received extra arguments: {kwargs.keys()}")

        self.cx = cx
        self.cy = 0.0
        self.r = radius

        # Sanity check
        assert radius > 0.0, "Radius must be positive"
        assert cx > radius, "Center x must be greater than radius"

        # Critical x value
        self.len_tengent_sq = cx ** 2 - radius ** 2
        self.len_tengent = math.sqrt(self.len_tengent_sq)
        self.critical_x = self.len_tengent_sq / self.cx
        self.arc_1 = math.atan2(radius * self.len_tengent / cx, self.critical_x - cx)

    def dist_1(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(x, dim=1, ord=2, keepdim=True)

    def dist_2(self, x: torch.Tensor) -> torch.Tensor:
        # segment 1: from source to obstacle along tangent line
        len_seg_1 = torch.tensor([[self.len_tengent]], device=x.device)

        # segment 3: from obstacle to destination along tangent line
        len_cx_dest_sq = (x - torch.tensor([[self.cx, 0.0]], device=x.device)).square().sum(dim=1, keepdim=True)
        len_seg_3 = torch.sqrt(len_cx_dest_sq - self.r ** 2)
        arc_2 = torch.acos(self.r / torch.sqrt(len_cx_dest_sq))
        arc_3 = torch.atan2(x[:, 1] - 0.0, x[:, 0] - self.cx).unsqueeze(1)

        # segment 2: along the arc
        len_seg_2 = (self.arc_1 - (arc_2 + arc_3)) * self.r

        # assert torch.all(len_seg_2 >= 0.0), "Segment 2 must be positive but got {}".format(len_seg_2)
        return len_seg_1 + len_seg_2 + len_seg_3

    def dist_3(self, x: torch.Tensor) -> torch.Tensor:
        # segment 1: from source to obstacle along tangent line
        len_seg_1 = torch.tensor([[self.len_tengent]], device=x.device)

        # segment 3: from obstacle to destination along tangent line
        len_cx_dest_sq = (x - torch.tensor([[self.cx, 0.0]], device=x.device)).square().sum(dim=1, keepdim=True)
        len_seg_3 = torch.sqrt(len_cx_dest_sq - self.r ** 2)
        arc_2 = torch.acos(self.r / torch.sqrt(len_cx_dest_sq))     # always >= 0
        arc_3 = torch.atan2(x[:, 1] - 0.0, x[:, 0] - self.cx).unsqueeze(1)      # should always <= 0

        # segment 2: along the arc
        len_seg_2 = ((-arc_2 + arc_3) + self.arc_1) * self.r

        # assert torch.all(len_seg_2 >= 0.0), "Segment 2 must be positive but got {}".format(len_seg_2)
        return len_seg_1 + len_seg_2 + len_seg_3
    
    def dist_4(self, x: torch.Tensor) -> torch.Tensor:
        vec_x_sq = x.square().sum(dim=1, keepdim=True)
        x0_4 = 4*x[:,[0]]
        x0_cx_4 = x0_4*self.cx
        b = self.len_tengent_sq-x0_cx_4-vec_x_sq
        a_c_4 = x0_cx_4*vec_x_sq*4
        inside_cx = (-b-(b.square()-a_c_4).sqrt())/x0_4

        # segment 1: from source to obstacle on arc
        obst_x = self.len_tengent_sq/(2*self.cx-inside_cx)
        vec_ic_ob_0 = obst_x-inside_cx
        obst_y_sq = -vec_ic_ob_0*obst_x
        len_seg_1 = (obst_x.square()+obst_y_sq).sqrt()

        # segment 2: along the inside arc
        inside_r_sq = vec_ic_ob_0.square()+obst_y_sq
        inside_r = inside_r_sq.sqrt()
        obst_y = obst_y_sq.sqrt()
        vec_ic_x_0 = x[:,[0]]-inside_cx
        arc = torch.acos((vec_ic_ob_0*vec_ic_x_0+obst_y*x[:,[1]].abs())/inside_r_sq) # (obst_x-inside_cx,obst_y), (x0-inside_cx,x1.abs())
        len_seg_2 = arc*inside_r

        return len_seg_1 + len_seg_2

    def forward(self, x: torch.Tensor, analytic_continuation: bool=True, keepdim: bool=False) -> torch.Tensor:
        assert x.shape[-1] == 2, "Input must have 2 elements in the last dimension"
        shape = x.shape[:-1]
        x = x.reshape(-1, 2)

        mask_invalid = (x - torch.tensor([[self.cx, 0.0]], device=x.device)).square().sum(dim=1) < self.r ** 2

        intersect_with_sphere = (x[:, 1] * self.cx) ** 2 < self.r ** 2 * (x[:, 0] ** 2 + x[:, 1] ** 2)
        mask_2 = (~mask_invalid) & (x[:, 0] > self.critical_x) & (x[:, 1] >= 0.0) & (intersect_with_sphere)
        mask_3 = (~mask_invalid) & (x[:, 0] > self.critical_x) & (x[:, 1] < 0.0) & (intersect_with_sphere)
        mask_1 = ~(mask_invalid | mask_2 | mask_3)

        mask_1 = mask_1.unsqueeze(-1)
        mask_2 = mask_2.unsqueeze(-1)
        mask_3 = mask_3.unsqueeze(-1)
        mask_invalid = mask_invalid.unsqueeze(-1)

        out = torch.where(mask_1, self.dist_1(x), torch.zeros_like(x[:, :1])) + \
              torch.where(mask_2, self.dist_2(x), torch.zeros_like(x[:, :1])) + \
              torch.where(mask_3, self.dist_3(x), torch.zeros_like(x[:, :1]))
        if analytic_continuation:
            out += torch.where(mask_invalid, self.dist_4(x), torch.zeros_like(x[:, :1]))

        out = torch.reshape(out, (*shape, -1))
        if not keepdim:
            out = out.squeeze(-1)
        return out

    def gradient(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs.requires_grad_(True)
        y = self.forward(inputs, analytic_continuation=False)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=inputs,
            grad_outputs=d_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def approx_ridge(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise RuntimeError(f"Cannot auto init ridge for this data because the ridge is unbounded. The ridge is ({self.cx+self.r}, 0) -> (+inf, 0). Please manually input the ridge info with regard to your training bounding box.")

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
