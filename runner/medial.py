#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Experiment runner for Medial dataset """

if __name__ == "__main__":
    import os, sys
    SCRIPTNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPTNAME)
    SCRIPTNAME = os.path.dirname(SCRIPTNAME)
    sys.path.append(SCRIPTNAME)

import argparse
import logging
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from runner.base import RunnerBase
from data.medial import data_factory

class RunnerMedial(RunnerBase):
    def __init__(
        self, *,
        conf_path: os.PathLike,
        mode: str="train",
        is_continue: bool=False,
        ckpt_name: Optional[str]=None,
        **kwargs,
    ):
        super().__init__(conf_path=conf_path)
        if kwargs:
            logging.warning(f"runner received extra arguments: {list(kwargs.keys())}")

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if ckpt_name is not None:
                latest_model_name = ckpt_name
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth':
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('found checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Code backup
        if mode.startswith("train"):
            self.file_backup()

        logging.info(f"end")

    def setup_dataset(self):
        dataset_config = self.conf.get_config('dataset')
        dataset = data_factory(**dataset_config).to(self.device)
        self.dataset = dataset
        return self.dataset

    def train(self):
        # Load various training-related configurations
        # Training process
        end_iter = self.conf.get_int('train.end_iter')
        save_freq = self.conf.get_int('train.save_freq')
        report_freq = self.conf.get_int('train.report_freq')
        val_freq = self.conf.get_int('train.val_freq')
        cd_freq = self.conf.get_int('train.cd_freq', default=-1)
        sample_size = self.conf.get_int('train.sample_size')
        batch_size = self.conf.get_int('train.batch_size', default=-1)
        learning_rate = self.conf.get_float('train.learning_rate')
        learning_rate_bem = self.conf.get_float('train.learning_rate_bem', default=learning_rate)
        learning_rate_ngp = self.conf.get_float('train.learning_rate_ngp', default=learning_rate)
        learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)

        # More Loss
        eikonal_weight = self.conf.get_float('train.eikonal_weight', default=0.0)
        laplacian_weight = self.conf.get_float('train.laplacian_weight', default=0.0)

        # BEM training config
        bem_freeze_until = self.conf.get_int('train.bem_freeze_until', default=-1)

        # Training region
        xmin = self.conf.get_float("train.xmin")
        xmax = self.conf.get_float("train.xmax")
        ymin = self.conf.get_float("train.ymin")
        ymax = self.conf.get_float("train.ymax")
        sample_size = self.conf.get_int('train.sample_size')

        writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))

        # Training loop
        for _ in tqdm(range(self.iter_step, end_iter)):
            if self.bem is not None:
                if bem_freeze_until < 0 or self.iter_step < bem_freeze_until:
                    self.bem.set_learnable(False)
                else:
                    self.bem.set_learnable(True)

            # Set learning rate
            learning_factor = self.get_learning_factor(warm_up_end=warm_up_end, end_iter=end_iter, learning_rate_alpha=learning_rate_alpha)
            self.update_learning_rate(
                lr_main=learning_rate * learning_factor,
                lr_bem=learning_rate_bem * learning_factor,
                lr_ingp=learning_rate_ngp * learning_factor,
            )

            # Prepare network input
            samples = self.get_perturbed_location(xmin, xmax, ymin, ymax, sample_size, sample_size, perturb=True).to(self.device)

            if batch_size > 0:
                samples = samples.split(batch_size)
            else:
                samples = [samples]
            basic_loss = torch.zeros(1, device=self.device)
            eikonal_loss = torch.zeros(1, device=self.device)
            for s in samples:
                # Prepare reference output
                reference = self.dataset.forward(s, keepdim=True)
                output = self.network.forward(s)
                basic_loss += torch.nn.functional.l1_loss(output, reference)
                if eikonal_weight > 0.0:
                    grad = self.network.gradient(s)
                    grad_norm = torch.linalg.vector_norm(grad, dim=-1)
                    eikonal_loss += torch.mean(torch.square(grad_norm - 1.0))
                else:
                    eikonal_loss += 0.0

            laplacian_loss = torch.zeros(1, device=self.device)
            if self.bem is not None:
                if laplacian_weight > 0.0: # Laplacian loss
                    laplacian_loss = self.bem.laplacian_loss()

            loss = basic_loss + \
                   eikonal_weight * eikonal_loss + \
                   laplacian_weight * laplacian_loss

            # Optimize
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            self.iter_step += 1

            # Loggings
            writer.add_scalar('train/loss', loss, self.iter_step)
            writer.add_scalar('train/loss basic', basic_loss, self.iter_step)
            writer.add_scalar('train/loss eikonal', eikonal_loss, self.iter_step)
            writer.add_scalar('train/loss laplacian', laplacian_loss, self.iter_step)

            if self.iter_step % save_freq == 0 or self.iter_step == bem_freeze_until or self.iter_step == end_iter:
                self.save_checkpoint()

            if self.iter_step % report_freq == 0:
                tqdm.write("Running Medial -> " + self.base_exp_dir)
                tqdm.write('iter:{:8>d} loss = {} lr = {}'.format(self.iter_step, loss.item(), learning_rate * learning_factor))

            if self.iter_step % val_freq == 0:
                self.visualise_output(resolution=512)

            if cd_freq > 0 and self.iter_step % cd_freq == 0 and self.bem is not None:
                cd = self.chamfer(samples=50000)
                with open(os.path.join(self.base_exp_dir, 'logs', 'chamfer.txt'), 'a') as f:
                    f.write(f"{self.iter_step} {cd}\n")
                writer.add_scalar('train/chamfer', cd, self.iter_step)

        logging.info("Training finished.")

    def visualise_output(self, resolution=64):
        xmin = self.conf.get_float("visual.xmin")
        xmax = self.conf.get_float("visual.xmax")
        ymin = self.conf.get_float("visual.ymin")
        ymax = self.conf.get_float("visual.ymax")
        levels = self.conf.get_list("visual.contour_levels", default=[])
        vmin = self.conf.get_float("visual.vmin")
        vmax = self.conf.get_float("visual.vmax")

        bbox = (xmin, xmax, ymin, ymax)
        # Generate Ground truth field for white dashed reference isolevel lines
        field_gt = self.export_field(
            lambda x: self.dataset.forward(x, keepdim=True),
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax,
            resolution=resolution,
        ).squeeze(-1)
        # Generate the network output
        field_out = self.export_field(
            self.network.forward,
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax,
            resolution=resolution,
        ).squeeze(-1)
        # Set up figure
        figure = plt.figure()
        ax: Axes = figure.add_axes([0, 0, 1, 1])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()

        # Background: Network output
        ax.imshow(field_out, origin="lower", cmap="winter", zorder=0, vmin=vmin, vmax=vmax, extent=bbox)
        # Layer 1: GT isolevel
        ax.contour(field_gt, levels=levels, colors="white", linestyles="dashed", linewidths=1.3, origin="lower", zorder=1, vmin=vmin, vmax=vmax, extent=bbox)
        # Layer 2: Network isolevel
        ax.contour(field_out, levels=levels, colors="orange", linestyles="solid", linewidths=1.3, origin="lower", zorder=2, vmin=vmin, vmax=vmax, extent=bbox)
        # Layer 3: Current sharp ridge
        if self.bem is not None:
            ridge_vertices = self.bem.vertices_all().detach().cpu().numpy()
            ridge_edges = self.bem.edges.detach().cpu().numpy()
            lines = []
            for f in ridge_edges:
                l = []
                for fi in f:
                    l.append(ridge_vertices[fi])
                l = np.array(l)
                lines.append(l)
            lc = LineCollection(lines, colors="orange", linestyles="solid", linewidths=1.3, zorder=3)
            ax.add_collection(lc)

        # Save figure
        os.makedirs(os.path.join(self.base_exp_dir, "validate", "png"), exist_ok=True)
        figure.savefig(os.path.join(self.base_exp_dir, "validate", "png", f"{self.iter_step:0>8d}.png"), bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=600)

        os.makedirs(os.path.join(self.base_exp_dir, "validate", "pdf"), exist_ok=True)
        figure.savefig(os.path.join(self.base_exp_dir, "validate", "pdf", f"{self.iter_step:0>8d}.pdf"), bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=600)

        plt.close(figure)

    @torch.no_grad()
    def chamfer(self, samples=100000):
        """
        Warning: This function should only work with Medial Axis Rectangle dataset.
        Only the Medial Axis Rectangle dataset has absolutely correct ground truth.
        """
        assert self.bem is not None, "Chamfer distance can only be computed when BEM is defined"
        gt_vs, gt_fs, _ = self.dataset.approx_ridge(segment_length=0.02)    # This value should be changed with the configuration file
        gt_vs = gt_vs.detach().cpu().numpy()
        gt_fs = gt_fs.detach().cpu().numpy()
        seq_for_gt = np.linspace(0.0, 1.0, samples // gt_fs.shape[0])
        gt_pts = []
        for f in gt_fs:
            gt_pts.append(
                gt_vs[f[0]][None, :] + (gt_vs[f[1]][None, :] - gt_vs[f[0]][None, :]) * seq_for_gt[:, None]
            )
        gt_pts = np.concatenate(gt_pts, axis=0)
        gt_pts = np.concatenate([
            gt_pts, np.zeros((gt_pts.shape[0], 1), dtype=np.float32)  # to 3d point.
        ], axis=1)

        vs = self.bem.vertices_all().detach().cpu().numpy()
        fs = self.bem.edges.detach().cpu().numpy()
        seq_for_out = np.linspace(0.0, 1.0, samples // fs.shape[0])
        pts = []
        for f in fs:
            pts.append(
                vs[f[0]][None, :] + (vs[f[1]][None, :] - vs[f[0]][None, :]) * seq_for_out[:, None]
            )
        pts = np.concatenate(pts, axis=0)
        pts = np.concatenate([
            pts, np.zeros((pts.shape[0], 1), dtype=np.float32)  # to 3d point.
        ], axis=1)

        # calculate the chamfer distance
        import open3d as o3d
        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
        pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        re1 = np.mean(np.asarray(pc1.compute_point_cloud_distance(pc2)))
        re2 = np.mean(np.asarray(pc2.compute_point_cloud_distance(pc1)))
        cd = (re1 + re2) / 2.0
        return cd.item()


def main():
    print("Hello Wooden")
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--is_continue", action="store_true", default=False)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ckpt_name", type=str, default=None)
    args = parser.parse_args()

    runner = RunnerMedial(conf_path=args.conf, is_continue=args.is_continue, mode=args.mode, ckpt_name=args.ckpt_name)

    if args.mode == "train":
        runner.train()
    elif args.mode == "validate":
        runner.visualise_output(resolution=args.resolution)

if __name__ == "__main__":
    main()
