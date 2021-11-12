# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


def lecun_normal_(x):
    return init.normal_(x, std=np.sqrt(1 / x.shape[1]))


def init_selu(x: nn.Linear):
    lecun_normal_(x.weight)
    init.zeros_(x.bias)
    return x


def init_linear(x: nn.Linear):
    init.xavier_uniform_(x.weight)
    init.zeros_(x.bias)
    return x


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def SNN(in_channels, out_channels, hidden_channels=[256, 256, 256, 256]):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        init_selu(nn.Linear(in_channels, hidden_channels[0])),
        nn.Sequential(
            *[
                Residual(
                    nn.Sequential(
                        init_selu(
                            nn.Linear(hidden_channels[idx - 1], hidden_channels[idx])
                        ),
                        nn.SELU(),
                    )
                )
                for idx in range(1, len(hidden_channels))
            ]
        ),
        init_linear(nn.Linear(hidden_channels[-1], out_channels)),
    )
