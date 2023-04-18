# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import torch
import torch.nn as nn
import torch.nn.functional as Functional
from typing import Union

from models.flame import FLAME


def kaiming_leaky_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.constant_(m.bias, 0.0)


def kaiming_selu_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
        torch.nn.init.constant_(m.bias, 0.0)


class MappingNetwork(nn.Module):
    def __init__(self, model_cfg, input_dim: int, hidden_dim: Union[int, list[int]], output_dim: int, hidden_layers: int = 2):
        super().__init__()
        
        self.cfg = model_cfg

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim for _ in range(hidden_layers)]       

        layers = [nn.BatchNorm1d(input_dim)] if self.cfg.batch_norm else []
        prev_dim = input_dim
        for dim in hidden_dim:
            
            layers.append(nn.Linear(prev_dim, dim))
            
            if self.cfg.batch_norm:
                layers.append(nn.BatchNorm1d(dim))

            layers.append(nn.SELU() if self.cfg.selu else nn.LeakyReLU(negative_slope=0.2))
            prev_dim = dim
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim[-1], output_dim)
        
        if self.cfg.selu:
            self.network.apply(kaiming_selu_init)
            self.output.apply(kaiming_selu_init)
        else:
            self.network.apply(kaiming_leaky_init)
            self.output.apply(kaiming_leaky_init)

        # with torch.no_grad():
        #     self.output.weight *= 0.25

    def forward(self, x):
        
        if not self.cfg.batch_norm:
            x = Functional.normalize(x)

        x = self.network(x)
        x = self.output(x)

        # TODO: Should this be here or elsewhere?
        if 0 < self.cfg.max_shape_code < float("inf"):
            x = torch.clip(x, min=-self.cfg.max_shape_code, max=self.cfg.max_shape_code)
        
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, hidden, model_cfg, device, regress=True):
        super().__init__()
        self.device = device
        self.cfg = model_cfg
        self.regress = regress

        if self.regress:
            self.regressor = MappingNetwork(model_cfg, z_dim, map_hidden_dim, map_output_dim, hidden).to(self.device)
        self.generator = FLAME(model_cfg).to(self.device)

    def forward(self, arcface):
        if self.regress:
            shape = self.regressor(arcface)
        else:
            shape = arcface

        prediction, _, _ = self.generator(shape_params=shape)

        return prediction, shape
