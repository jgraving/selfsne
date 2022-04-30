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
import torch.nn.functional as F
import numpy as np


def lecun_normal_(x, mode="fan_in"):
    return init.kaiming_normal_(x, mode=mode, nonlinearity="linear")


def init_selu(x):
    lecun_normal_(x.weight)
    if hasattr(x, "bias"):
        if x.bias is not None:
            init.zeros_(x.bias)
    return x


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ParametricResidual(nn.Module):
    def __init__(self, in_channels, out_channels, module):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.module = module

    def forward(self, x):
        return self.proj(x) + self.module(x)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs,
        )

    def forward(self, x):
        return super().forward(x)[..., : -self.padding[0]]


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=((kernel_size - 1) * dilation // 2),
            **kwargs,
        )


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean((-1, -2))


def TCN(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    n_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    normalize_input=False,
    batch_norm=False,
):
    """Temporal Convolution Network (TCN)"""
    conv1d = CausalConv1d if causal else Conv1d
    return nn.Sequential(
        nn.BatchNorm1d(in_channels) if normalize_input else nn.Identity(),
        PadShift() if causal and causal_shift else nn.Identity(),
        init_selu(nn.Conv1d(in_channels, hidden_channels, 1)),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            *[
                                Residual(
                                    nn.Sequential(
                                        nn.BatchNorm1d(hidden_channels)
                                        if batch_norm
                                        else nn.Identity(),
                                        init_selu(
                                            conv1d(
                                                hidden_channels,
                                                hidden_channels,
                                                kernel_size=kernel_size,
                                                dilation=dilation,
                                            )
                                        ),
                                        nn.SELU(),
                                    )
                                )
                                for dilation in 2 ** np.arange(n_layers)
                            ]
                        )
                    )
                    for _ in range(n_blocks)
                ]
            )
        ),
        nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity(),
        init_selu(nn.Conv1d(hidden_channels, out_channels, 1)),
    )


def TCN2d(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    n_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    normalize_input=False,
    batch_norm=False,
):
    """Temporal Convolution Network (TCN)"""
    conv = nn.Conv2d
    return nn.Sequential(
        nn.BatchNorm2d(in_channels) if normalize_input else nn.Identity(),
        init_selu(nn.Conv2d(in_channels, hidden_channels, (1, 1))),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            *[
                                Residual(
                                    nn.Sequential(
                                        nn.BatchNorm2d(hidden_channels)
                                        if batch_norm
                                        else nn.Identity(),
                                        init_selu(
                                            conv(
                                                hidden_channels,
                                                hidden_channels,
                                                kernel_size=(1, kernel_size),
                                                dilation=(1, dilation),
                                                padding=(
                                                    0,
                                                    ((kernel_size - 1) * dilation // 2),
                                                ),
                                            )
                                        ),
                                        nn.SELU(),
                                    )
                                )
                                for dilation in 2 ** np.arange(n_layers)
                            ]
                        )
                    )
                    for _ in range(n_blocks)
                ]
            )
        ),
        nn.BatchNorm2d(hidden_channels) if batch_norm else nn.Identity(),
        init_selu(nn.Conv2d(hidden_channels, out_channels, (1, 1))),
    )


def TCN3d(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    n_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    normalize_input=False,
    residual=False,
    batch_norm=False,
):
    """Temporal Convolution Network (TCN)"""
    conv = nn.Conv3d
    return nn.Sequential(
        nn.BatchNorm3d(in_channels) if normalize_input else nn.Identity(),
        init_selu(nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            *[
                                Residual(
                                    nn.Sequential(
                                        nn.BatchNorm3d(hidden_channels)
                                        if batch_norm
                                        else nn.Identity(),
                                        init_selu(
                                            conv(
                                                hidden_channels,
                                                hidden_channels,
                                                kernel_size=(1, 1, kernel_size),
                                                dilation=(1, 1, dilation),
                                                padding=(
                                                    0,
                                                    0,
                                                    ((kernel_size - 1) * dilation // 2),
                                                ),
                                            )
                                        ),
                                        nn.SELU(),
                                    )
                                )
                                for dilation in 2 ** np.arange(n_layers)
                            ]
                        )
                    )
                    for _ in range(n_blocks)
                ]
            )
        ),
        nn.BatchNorm3d(hidden_channels) if batch_norm else nn.Identity(),
        init_selu(nn.Conv3d(hidden_channels, out_channels, (1, 1, 1))),
    )


def ResNet2d(
    in_channels,
    out_channels,
    hidden_channels=64,
    n_layers=4,
    n_blocks=4,
    global_pooling=True,
    normalize_input=False,
    batch_norm=False,
):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, affine=False, momentum=None)
        if normalize_input
        else nn.Identity(),
        init_selu(nn.Conv2d(in_channels, hidden_channels, 7, stride=2, padding=3)),
        nn.SELU(),
        nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            *[
                                Residual(
                                    nn.Sequential(
                                        nn.BatchNorm2d(hidden_channels)
                                        if batch_norm
                                        else nn.Identity(),
                                        init_selu(
                                            nn.Conv2d(
                                                hidden_channels,
                                                hidden_channels,
                                                kernel_size=3,
                                                padding=1,
                                            )
                                        ),
                                        nn.SELU(),
                                    )
                                )
                                for _ in range(n_layers)
                            ]
                        )
                    ),
                    nn.AvgPool2d(3, 2, 1),
                )
                for _ in range(n_blocks)
            ]
        ),
        nn.BatchNorm2d(hidden_channels) if batch_norm else nn.Identity(),
        init_selu(nn.Conv2d(hidden_channels, out_channels, 1)),
        GlobalAvgPool2d() if global_pooling else nn.Identity(),
    )


def MLP(
    in_channels,
    out_channels,
    hidden_channels=256,
    n_layers=4,
    normalize_input=False,
    batch_norm=False,
):
    net = nn.Sequential(
        nn.BatchNorm1d(in_channels) if normalize_input else nn.Identity(),
        init_selu(nn.Linear(in_channels, hidden_channels)),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            nn.BatchNorm1d(hidden_channels)
                            if batch_norm
                            else nn.Identity(),
                            init_selu(nn.Linear(hidden_channels, hidden_channels)),
                            nn.SELU(),
                        )
                    )
                    for _ in range(n_layers)
                ]
            )
        ),
        nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity(),
        init_selu(nn.Linear(hidden_channels, out_channels)),
    )
    return (
        Residual(net)
        if in_channels == out_channels
        else ParametricResidual(in_channels, out_channels, net)
    )
