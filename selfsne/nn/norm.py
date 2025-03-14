# nn/norm.py
# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F

from selfsne.nn.layers import Residual  # For the prenormalization wrappers

__all__ = [
    "ImageNetNorm",
    "InputNorm1d",
    "InputNorm2d",
    "InputNorm3d",
    "DyT",
    "PreDyT",
    "PreLayerNorm",
    "PreRMSNorm",
]


class ImageNetNorm(nn.Module):
    """
    Normalizes input images using ImageNet's mean and standard deviation.
    """

    def __init__(self):
        super().__init__()
        loc = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        scale = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, x):
        return (x - self.loc) / self.scale


class InputNorm1d(nn.BatchNorm1d):
    """
    Batch normalization for 1D inputs with custom handling based on input dimensionality.
    """

    def __init__(self, num_features: int, eps: float = 1e-05, momentum=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)  # Updates running stats
        if x.dim() == 2:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        elif x.dim() == 3:
            running_mean = self.running_mean.view(1, -1, 1)
            running_var = self.running_var.view(1, -1, 1)
            x_norm = (x - running_mean) / torch.sqrt(running_var + self.eps)
        else:
            raise ValueError(f"Unexpected input dimension {x.dim()}, expected 2 or 3.")
        return x_norm


class InputNorm2d(nn.BatchNorm2d):
    """
    Batch normalization for 2D inputs with custom handling.
    """

    def __init__(self, num_features: int, eps: float = 1e-05, momentum=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        x_norm = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1, 1) + self.eps
        )
        return x_norm


class InputNorm3d(nn.BatchNorm3d):
    """
    Batch normalization for 3D inputs with custom handling.
    """

    def __init__(self, num_features: int, eps: float = 1e-05, momentum=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        x_norm = (x - self.running_mean.view(1, -1, 1, 1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1, 1, 1) + self.eps
        )
        return x_norm


class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) layer as a drop-in replacement for normalization layers.
    Applies an element-wise tanh to the input after scaling by a learnable alpha,
    then applies per-channel affine transformation (gamma and beta).

    DyT(x) = gamma * tanh(alpha * x) + beta
    """

    def __init__(self, num_features, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Assumes channel dimension is the last dimension.
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


def PreDyT(in_features, module, init_alpha=0.5):
    """
    Wraps a module with a prenormalization step using DyT.
    """
    return Residual(nn.Sequential(DyT(in_features, init_alpha), module))


def PreLayerNorm(in_features, module):
    """
    Wraps a module with a prenormalization step using Layer Normalization.
    """
    return Residual(nn.Sequential(nn.LayerNorm(in_features), module))


def PreRMSNorm(in_features, module):
    """
    Wraps a module with a prenormalization step using RMS Normalization.
    Note: Ensure that nn.RMSNorm is defined or replace with a custom RMSNorm implementation.
    """
    return Residual(nn.Sequential(nn.RMSNorm(in_features), module))
