# nn/layers.py
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

from torch import nn
import torch.nn.functional as F

from selfsne.nn.init import init_linear
from selfsne.utils import stop_gradient

__all__ = [
    "Residual",
    "ParametricResidual",
    "Residual1d",
    "Residual2d",
    "Lambda",
    "StopGradient",
]


class Residual(nn.Module):
    def __init__(self, module, residual=nn.Identity()):
        super().__init__()
        self.module = module
        self.residual = residual

    def forward(self, x):
        return self.residual(x) + self.module(x)


def ParametricResidual(in_features, out_features, module):
    return Residual(
        module,
        init_linear(nn.Linear(in_features, out_features)),
    )


def Residual1d(in_features, out_features, module):
    residual = (
        init_linear(nn.Linear(in_features, out_features))
        if in_features != out_features
        else nn.Identity()
    )
    return Residual(module, residual)


def Residual2d(in_channels, out_channels, module):
    residual = (
        init_linear(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        if in_channels != out_channels
        else nn.Identity()
    )
    return Residual(module, residual)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class StopGradient(nn.Module):
    def forward(self, x):
        return stop_gradient(x)
