#!/usr/bin/env false
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pyhocon import ConfigTree, ConfigFactory
import random
import shutil
import torch
from typing import Optional, Tuple, List, Dict
import net
from net import BEMquery, MLP, SharpNet, Encoder
from data import Dataset

""" Base experiment runner class """

class RunnerBase(object, metaclass=ABCMeta):

    # Configurations
    conf: ConfigTree
    conf_path: os.PathLike
    base_exp_dir: os.PathLike
    random_seed: int

    # Network
    bem: Optional[BEMquery]
    mlp: MLP
    input_encoder: Optional[Encoder]
    feature_encoder: Optional[Encoder]
    network: SharpNet
    optimiser: torch.optim.Optimizer

    # Other
    dataset: Dataset
    device: torch.device
    iter_step: int
    _optimiser_parameters: List[str]

    def __init__(self, conf_path: os.PathLike):
        """ Basic initialisation of the experiment runner """
        super().__init__()

        self.iter_step = 0

        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Using device: {self.device}")

        # Set up configuration
        self.conf_path = conf_path
        with open(conf_path, "rt") as f:
            conf_text = f.read()
            self.conf = ConfigFactory.parse_string(conf_text)

        self.base_exp_dir = self.conf.get_string("general.base_exp_dir")
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.random_seed = self.conf.get_int("general.random_seed", default=2025)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set up dataset
        self.dataset = self.setup_dataset().to(self.device)

        # Set up neural network
        self.bem = self._setup_bem()
        self.input_encoder = self._setup_input_encoder()
        self.feature_encoder = self._setup_feature_encoder()
        self.mlp = self._setup_mlp()
        self.network = SharpNet(mlp=self.mlp, bem=self.bem, input_enc=self.input_encoder, feat_enc=self.feature_encoder)

        # Set up optimiser
        self.optimiser = self._setup_optim()


    @abstractmethod
    def setup_dataset(self) -> Dataset:
        pass

    def _setup_bem(self) -> Optional[BEMquery]:
        if not "network.bem" in self.conf:
            logging.info("BEM Module configuration absent. No BEM module instantiated.")
            self.bem = None
        else:
            bem_auto = self.conf.get_bool("network.bem.auto")
            bem_radius = self.conf.get_float("network.bem.radius", default=-1.0)
            bem_eps = self.conf.get_float("network.bem.eps", default=1e-16)
            bem_split_feature = self.conf.get_bool("network.bem.split_feature", default=False)

            if bem_auto:
                logging.info("BEM Module is instantiated automatically")
                bem_segment_length = self.conf.get_float("network.bem.seglen", default=-1.0)
                bem_vertices, bem_edges, _ = self.dataset.approx_ridge(segment_length=bem_segment_length)
            else:
                logging.info("BEM Module is instantiated manually")
                bem_vertices = self.conf.get_list("network.bem.vertices", default=[])
                bem_edges = self.conf.get_list("network.bem.edges", default=[])

            bem = BEMquery(vertices=bem_vertices, edges=bem_edges, nonmanifolds=None, eps=bem_eps, radius=bem_radius, once_bias=False, split_feature=bem_split_feature)
            self.bem = bem.to(self.device)
        return self.bem

    def _setup_input_encoder(self) -> Optional[Encoder]:
        d_in = self.conf.get_int("network.d_in")
        if not "network.input_encoding" in self.conf:
            logging.info("No input encoder configuration, input will be passed through without encoding.")
            self.input_encoder = None
        else:
            flavour = self.conf.get_string("network.input_encoding.flavour")
            if flavour.casefold() == "pe":
                # Configuration for positional encoder
                logging.info("Input encoder: Positional Encoder")
                multires = self.conf.get_int("network.input_encoding.multires")
                if multires > 0:
                    input_encoder_config = self.conf.get_config("network.input_encoding")
                    input_encoder_config.pop("flavour")
                    input_encoder_config.pop("multires")    # Must succeed because multires is extracted above
                    input_encoder = net.encoding.PositionalEncoder(input_dims=d_in, num_freqs=multires, **input_encoder_config)
                    self.input_encoder = input_encoder.to(self.device)
                else:
                    logging.warning("Input encoder disabled since multires <= 0. Should omit the entire input_encoding block in config.")
                    self.input_encoder = None
            elif flavour.casefold() == "ingp":
                # Configuration for INGP encoder
                scale = self.conf.get_float("network.input_encoding.scale", default=1.0)
                shift = self.conf.get_list("network.input_encoding.shift", default=[0.0, 0.0])
                input_encoder_config = self.conf.get_config("network.input_encoding")
                try:
                    input_encoder_config.pop("flavour")
                    if "scale" in input_encoder_config:
                        input_encoder_config.pop("scale")
                    if "shift" in input_encoder_config:
                        input_encoder_config.pop("shift")
                finally:
                    self.input_encoder = net.encoding.INGPEncoder(scale=scale, shift=shift, n_input_dims=d_in, encoding_config=input_encoder_config, dtype=torch.float32)
                logging.info("Input encoder: InstantNGP Encoder")
            else:
                raise RuntimeError("Unknown input encoder flavour specified in configuration:", flavour)
        return self.input_encoder

    def _setup_feature_encoder(self) -> Optional[Encoder]:
        if self.bem is None:
            logging.info("No BEM module, therefore no feature encoder instantiated.")
            self.feature_encoder = None
        elif not "network.feature_encoding" in self.conf:
            logging.info("No feature encoder configuration, features will be passed through without encoding.")
            self.feature_encoder = None
        else:
            # Feature encoder is always positional encoder if exists
            d_feat = self.bem.output_dim()
            multires = self.conf.get_int("network.feature_encoding.multires")
            if multires > 0:
                logging.info("Feature encoder: Positional Encoder")
                feature_encoder_config = self.conf.get_config("network.feature_encoding")
                feature_encoder_config.pop("multires")    # Must succeed because multires is extracted above
                feature_encoder = net.encoding.PositionalEncoder(input_dims=d_feat, num_freqs=multires, **feature_encoder_config)
                self.feature_encoder = feature_encoder.to(self.device)
            else:
                logging.warning("Feature encoder disabled since multires <= 0. Should omit the entire feature_encoding block in config.")
                self.feature_encoder = None
        return self.feature_encoder

    def _setup_mlp(self) -> MLP:
        mlp_config = self.conf.get_config("network.mlp")

        if self.input_encoder is not None:
            d_in = self.input_encoder.output_dim()
        else:
            d_in = self.conf.get_int("network.d_in")

        if self.bem is None:
            d_feat = 0
        elif self.feature_encoder is None:
            d_feat = self.bem.output_dim()
        else:
            d_feat = self.feature_encoder.output_dim()

        d_out = self.conf.get_int("network.d_out")
        mlp = MLP(d_in=d_in, d_out=d_out, d_feat=d_feat, **mlp_config)
        self.mlp = mlp.to(self.device)
        return self.mlp

    def _setup_optim(self) -> torch.optim.Optimizer:
        self._optimiser_parameters = []
        params_to_train = [
            {"params": list(self.mlp.parameters())}
        ]
        self._optimiser_parameters.append("mlp")

        if self.bem is not None:
            params_to_train.append({"params": list(self.bem.parameters())})
            self._optimiser_parameters.append("bem")

        if self.input_encoder is not None:
            params_to_train.append({"params": list(self.input_encoder.parameters())})
            self._optimiser_parameters.append("input_enc")

        if self.feature_encoder is not None:
            params_to_train.append({"params": list(self.feature_encoder.parameters())})
            self._optimiser_parameters.append("feature_enc")

        optim = torch.optim.Adam(params_to_train, lr=0.00)
        self.optimiser = optim
        return self.optimiser

    def update_learning_rate(self, lr_main: float, lr_bem: float = 0.0, lr_ingp: float = 0.0):
        for param_name, param_group in zip(self._optimiser_parameters, self.optimiser.param_groups):
            if param_name == "mlp":
                param_group['lr'] = lr_main
            elif param_name == "bem":
                param_group['lr'] = lr_bem
            elif param_name == "input_enc":
                assert self.input_encoder is not None
                if isinstance(self.input_encoder, net.encoding.PositionalEncoder):
                    param_group['lr'] = lr_main
                elif isinstance(self.input_encoder, net.encoding.INGPEncoder):
                    param_group['lr'] = lr_ingp
                else:
                    raise RuntimeError(f"Unknown input encoder type {type(self.input_encoder)} when updating learning rate.")
            elif param_name == "feature_enc":
                param_group['lr'] = lr_main
            else:
                raise RuntimeError(f"Unknown parameter group name {param_name} when updating learning rate.")
            
    def load_checkpoint(self, checkpoint_name: os.PathLike):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device, weights_only=False)
        self.iter_step = checkpoint["iter_step"]
        self.mlp.load_state_dict(checkpoint["mlp"])
        if self.bem is not None:
            self.bem.load_state_dict(checkpoint["bem"])
        if self.input_encoder is not None:
            self.input_encoder.load_state_dict(checkpoint["input_encoder"])
        if self.feature_encoder is not None:
            self.feature_encoder.load_state_dict(checkpoint["feature_encoder"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        logging.info(f"Checkpoint loaded: {checkpoint_name}")

    def save_checkpoint(self):
        checkpoint = {
            "iter_step": self.iter_step,
            "mlp": self.mlp.state_dict(),
            "optimiser": self.optimiser.state_dict(),
        }
        if self.bem is not None:
            checkpoint["bem"] = self.bem.state_dict()
        if self.input_encoder is not None:
            checkpoint["input_encoder"] = self.input_encoder.state_dict()
        if self.feature_encoder is not None:
            checkpoint["feature_encoder"] = self.feature_encoder.state_dict()
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def file_backup(self):
        dir_lis = self.conf.get_list("general.recording", default=[])
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name.endswith(".py"):
                    shutil.copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        shutil.copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def get_learning_factor(self, warm_up_end: int, end_iter: int, learning_rate_alpha: float) -> float:
        if self.iter_step < warm_up_end:
            learning_factor = self.iter_step / warm_up_end
        else:
            alpha = learning_rate_alpha
            progress = (self.iter_step - warm_up_end) / (end_iter - warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        return learning_factor

    @staticmethod
    def get_perturbed_location(xmin, xmax, ymin, ymax, xsample, ysample, perturb:bool=False):
        # Divide [xmin, xmax] into xsample equal regions
        x_coords = torch.linspace(xmin, xmax, xsample+1)
        x_coords = (x_coords[1:] + x_coords[:-1]) / 2.0
        # Do the same to [ymin, ymax]
        y_coords = torch.linspace(ymin, ymax, ysample+1)
        y_coords = (y_coords[1:] + y_coords[:-1]) / 2.0
        if perturb:
            x_span = (xmax - xmin) / xsample
            x_perturb = (torch.rand_like(x_coords) - 0.5) * x_span
            x_coords += x_perturb
            y_span = (ymax - ymin) / ysample
            y_perturb = (torch.rand_like(y_coords) - 0.5) * y_span
            y_coords += y_perturb

        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
        return pts

    def export_field(
        self,
        query_func: Callable[[torch.Tensor], torch.Tensor],
        xmin: float, xmax: float, ymin: float, ymax: float,
        resolution: int=64,
        dim: Optional[int]=None,
    ) -> np.ndarray:
        N = 64
        X = torch.linspace(xmin, xmax, resolution).split(N)
        Y = torch.linspace(ymin, ymax, resolution).split(N)
        dim = self.dataset.output_dim() if dim is None else dim

        u = np.empty([resolution, resolution, dim])
        for yi, ys in enumerate(Y):
            for xi, xs in enumerate(X):
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                xx = xx.reshape(-1, 1)
                yy = yy.reshape(-1, 1)
                pts = torch.cat([xx, yy], dim=1).to(self.device)
                val = query_func(pts)
                val = val.reshape(len(ys), len(xs), dim).detach().cpu().numpy()
                u[yi*N:yi*N+len(ys), xi*N:xi*N+len(xs), :] = val
        return u