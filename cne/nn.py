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


class TwinEncoder(nn.Module):
    def __init__(
        self,
        encoder=nn.Identity(),
        noise_a=nn.Identity(),
        noise_b=nn.Identity(),
        encoder_a=nn.Identity(),
        encoder_b=nn.Identity(),
        subsampler=nn.Identity(),
    ):
        super().__init__()
        self.encoder = encoder
        self.noise_a = noise_a
        self.noise_b = noise_b
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.subsampler = subsampler

    def forward(self, x):
        x_a = self.noise_a(x)
        x_b = self.noise_b(x)
        z_a = self.encoder_a(self.encoder(x_a))
        z_b = self.encoder_b(self.encoder(x_b))
        return self.subsampler([z_a, z_b])


def lecun_normal_(x):
    return init.normal_(x, std=np.sqrt(1 / x.view(x.shape[0], -1).shape[-1]))


def init_selu(x: nn.Linear):
    lecun_normal_(x.weight)
    init.zeros_(x.bias)
    return x


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # use average to preserve self-normalization
        return 0.5 * (x + self.module(x))


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


class GlobalAveragePooling2d(nn.Module):
    def forward(self, x):
        return x.mean((-1, -2))


class FlipAxis(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.flip(self.dim)


def FlipSequence():
    return FlipAxis(-1)


class SequenceSubsampler(nn.Module):
    def forward(self, x):
        x_a, x_b = x
        if self.training:
            max_window = x_a.shape[-1] // 2
            idx = torch.randint(x_a.shape[-1] - max_window, size=(x_a.shape[0],))
            offset = torch.randint(low=1, high=max_window, size=(x_a.shape[0],))
            batch = torch.arange(x_a.shape[0])
            return x_a[batch, ..., idx], x_b[batch, ..., idx + offset]
        else:
            idx = x_a.shape[-1] // 2
            return x_a[..., idx], x_b[..., idx]


def TCN(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    n_layers=4,
    n_blocks=4,
    causal=False,
    normalize_input=False,
):
    """Temporal Convolution Network (TCN)"""
    conv1d = CausalConv1d if causal else Conv1d
    return nn.Sequential(
        nn.BatchNorm1d(in_channels, affine=False, momentum=None)
        if normalize_input
        else nn.Identity(),
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
        init_selu(nn.Conv1d(hidden_channels, out_channels, 1)),
    )


def ResNet2d(
    in_channels,
    out_channels,
    hidden_channels=64,
    n_layers=4,
    n_blocks=4,
    global_pooling=True,
    normalize_input=False,
):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels, affine=False, momentum=None)
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
                    nn.MaxPool2d(2, 2),
                )
                for _ in range(n_blocks)
            ]
        ),
        init_selu(nn.Conv2d(hidden_channels, out_channels, 1)),
        nn.SELU(),
        GlobalAveragePooling2d() if global_pooling else nn.Identity(),
    )


def SNN(
    in_channels, out_channels, hidden_channels=256, n_layers=4, normalize_input=False
):
    """Self-normalizing Neural Network (SNN)"""
    return nn.Sequential(
        nn.BatchNorm1d(in_channels, affine=False, momentum=None)
        if normalize_input
        else nn.Identity(),
        init_selu(nn.Linear(in_channels, hidden_channels)),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            init_selu(nn.Linear(hidden_channels, hidden_channels)),
                            nn.SELU(),
                        )
                    )
                    for _ in range(n_layers)
                ]
            )
        ),
        init_selu(nn.Linear(hidden_channels, out_channels)),
    )
