#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Experiment runner for Belhe dataset """

if __name__ == "__main__":
    import os, sys
    SCRIPTNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPTNAME)
    SCRIPTNAME = os.path.dirname(SCRIPTNAME)
    sys.path.append(SCRIPTNAME)

import argparse
import logging
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from runner.base import RunnerBase
from data.belhe import Belhe

class RunnerBelhe(RunnerBase):
    def __init__(
        self, *,
        conf_path: os.PathLike,
        mode: str="train",
        is_continue: bool=False,
        **kwargs,
    ):
        super().__init__(conf_path=conf_path)
        if kwargs:
            logging.warning(f"runner received extra arguments: {list(kwargs.keys())}")

        # Load checkpoint
        latest_model_name = None
        if is_continue:
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

    def setup_dataset(self) -> Belhe:
        dataset = Belhe()
        self.dataset = dataset.to(self.device)
        return self.dataset

    def train(self):
        # Load various training-related configurations
        # Training process
        end_iter = self.conf.get_int('train.end_iter')
        save_freq = self.conf.get_int('train.save_freq')
        report_freq = self.conf.get_int('train.report_freq')
        val_freq = self.conf.get_int('train.val_freq')
        learning_rate = self.conf.get_float('train.learning_rate')
        learning_rate_bem = self.conf.get_float('train.learning_rate_bem', default=learning_rate)
        learning_rate_ngp = self.conf.get_float('train.learning_rate_ngp', default=learning_rate)
        learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)

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
                self.bem.set_learnable(False)

            # Set learning rate
            learning_factor = self.get_learning_factor(warm_up_end=warm_up_end, end_iter=end_iter, learning_rate_alpha=learning_rate_alpha)
            self.update_learning_rate(
                lr_main=learning_rate * learning_factor,
                lr_bem=learning_rate_bem * learning_factor,
                lr_ingp=learning_rate_ngp * learning_factor,
            )

            samples = self.get_perturbed_location(xmin, xmax, ymin, ymax, sample_size, sample_size, perturb=True).to(self.device)
            reference = self.dataset.forward(samples, keepdim=True)
            output = self.network.forward(samples)
            loss = torch.nn.functional.l1_loss(output, reference)

            # Optimize
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            self.iter_step += 1

            # Loggings
            writer.add_scalar('train/loss', loss, self.iter_step)

            if self.iter_step % save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % report_freq == 0:
                tqdm.write("Running Belhe -> " + self.base_exp_dir)
                tqdm.write('iter:{:8>d} loss = {} lr = {}'.format(self.iter_step, loss, learning_rate * learning_factor))

            if self.iter_step % val_freq == 0:
                self.visualise_output(resolution=512)

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

        # Save figure
        os.makedirs(os.path.join(self.base_exp_dir, "validate", "png"), exist_ok=True)
        figure.savefig(os.path.join(self.base_exp_dir, "validate", "png", f"{self.iter_step:0>8d}.png"), bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=600)

        os.makedirs(os.path.join(self.base_exp_dir, "validate", "pdf"), exist_ok=True)
        figure.savefig(os.path.join(self.base_exp_dir, "validate", "pdf", f"{self.iter_step:0>8d}.pdf"), bbox_inches="tight", pad_inches=0.0, transparent=True, dpi=600)

        plt.close(figure)

def main():
    print("Hello Wooden")
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--is_continue", action="store_true", default=False)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    runner = RunnerBelhe(conf_path=args.conf, is_continue=args.is_continue, mode=args.mode)

    if args.mode == "train":
        runner.train()
    elif args.mode == "validate":
        runner.visualise_output(resolution=args.resolution)

if __name__ == "__main__":
    main()
