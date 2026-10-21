# nn/conv.py
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
import numpy as np

from selfsne.nn.init import init_linear
from selfsne.nn.layers import Residual

__all__ = [
    "PadShift",
    "CausalConv1d",
    "Conv1d",
    "TCN",
    "TCN2d",
    "TCN3d",
    "ResNet2d",
]


class PadShift(nn.Module):
    def forward(self, x):
        # Pads one element to the left and removes the last element along the last dimension
        return F.pad(x, pad=(1, 0))[..., :-1]


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        # Compute the total padding required to achieve a causal convolution.
        pad = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=pad,
            **kwargs,
        )

    def forward(self, x):
        # Forward pass produces extra timesteps at the end due to padding.
        out = super().forward(x)
        # Output index i corresponds to input time i * stride. Keep only
        # positions belonging to the input, including the final partial stride.
        length = (x.shape[-1] + self.stride[0] - 1) // self.stride[0]
        return out[..., :length]


class Conv1d(nn.Conv1d):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, **kwargs
    ):
        total = (kernel_size - 1) * dilation
        self._temporal_padding = (total // 2, total - total // 2)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=stride,
            padding=0,
            **kwargs,
        )

    def forward(self, x):
        mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
        return super().forward(F.pad(x, self._temporal_padding, mode=mode))


class _TemporalPadding:
    """Convolve along the last axis while preserving all spatial dimensions."""

    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, causal=False
    ):
        total = (kernel_size - 1) * dilation
        self._temporal_padding = (
            (total, 0) if causal else (total // 2, total - total // 2)
        )
        spatial = (1,) * (self._dimensions - 1)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(*spatial, kernel_size),
            dilation=(*spatial, int(dilation)),
            padding=0,
        )

    def forward(self, x):
        return super().forward(F.pad(x, self._temporal_padding))


class _TemporalConv2d(_TemporalPadding, nn.Conv2d):
    _dimensions = 2


class _TemporalConv3d(_TemporalPadding, nn.Conv3d):
    _dimensions = 3


def _validate_causal_normalization(causal, batch_norm):
    if causal and batch_norm:
        raise ValueError(
            "batch_norm=True uses future timesteps and cannot be combined with causal=True."
        )


def TCN(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    num_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    batch_norm=False,
):
    """
    Temporal Convolution Network (TCN) for 1D sequences.

    Causal networks exclude future timesteps; batch_norm must be False.
    """
    _validate_causal_normalization(causal, batch_norm)
    conv1d = CausalConv1d if causal else Conv1d
    layers = []
    if causal and causal_shift:
        layers.append(PadShift())
    layers.append(init_linear(nn.Conv1d(in_channels, hidden_channels, kernel_size=1)))
    layers.append(nn.SELU())

    block_list = []
    for _ in range(n_blocks):
        layer_list = []
        for dilation in 2 ** np.arange(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity(),
                init_linear(
                    conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    )
                ),
                nn.SELU(),
            )
            layer_list.append(layer)
        block_list.append(Residual(nn.Sequential(*layer_list)))
    layers.append(Residual(nn.Sequential(*block_list)))
    layers.append(nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity())
    layers.append(init_linear(nn.Conv1d(hidden_channels, out_channels, kernel_size=1)))
    return nn.Sequential(*layers)


def TCN2d(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    num_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    batch_norm=False,
):
    """
    Temporal Convolution Network (TCN) over the last axis of 2D data.

    Causal networks exclude future timesteps; batch_norm must be False.
    """
    _validate_causal_normalization(causal, batch_norm)
    conv = _TemporalConv2d
    layers = []
    if causal and causal_shift:
        layers.append(PadShift())
    layers.append(init_linear(nn.Conv2d(in_channels, hidden_channels, (1, 1))))
    layers.append(nn.SELU())

    block_list = []
    for _ in range(n_blocks):
        layer_list = []
        for dilation in 2 ** np.arange(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(hidden_channels) if batch_norm else nn.Identity(),
                init_linear(
                    conv(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        causal=causal,
                    )
                ),
                nn.SELU(),
            )
            layer_list.append(layer)
        block_list.append(Residual(nn.Sequential(*layer_list)))
    layers.append(Residual(nn.Sequential(*block_list)))
    layers.append(nn.BatchNorm2d(hidden_channels) if batch_norm else nn.Identity())
    layers.append(init_linear(nn.Conv2d(hidden_channels, out_channels, (1, 1))))
    return nn.Sequential(*layers)


def TCN3d(
    in_channels,
    out_channels,
    kernel_size=3,
    hidden_channels=64,
    num_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    residual=False,
    batch_norm=False,
):
    """
    Temporal Convolution Network (TCN) over the last axis of 3D data.

    Causal networks exclude future timesteps; batch_norm must be False.
    Set residual=True to enable skip connections around blocks.
    """
    _validate_causal_normalization(causal, batch_norm)
    conv = _TemporalConv3d
    wrap = Residual if residual else lambda module: module
    layers = []
    if causal and causal_shift:
        layers.append(PadShift())
    layers.append(init_linear(nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))))
    layers.append(nn.SELU())

    block_list = []
    for _ in range(n_blocks):
        layer_list = []
        for dilation in 2 ** np.arange(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm3d(hidden_channels) if batch_norm else nn.Identity(),
                init_linear(
                    conv(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        causal=causal,
                    )
                ),
                nn.SELU(),
            )
            layer_list.append(layer)
        block_list.append(wrap(nn.Sequential(*layer_list)))
    layers.append(wrap(nn.Sequential(*block_list)))
    layers.append(nn.BatchNorm3d(hidden_channels) if batch_norm else nn.Identity())
    layers.append(init_linear(nn.Conv3d(hidden_channels, out_channels, (1, 1, 1))))
    return nn.Sequential(*layers)


def ResNet2d(
    in_channels,
    out_channels,
    hidden_channels=64,
    num_layers=2,
    num_blocks=4,
    global_pooling=True,
    batch_norm=False,
    input_stride=2,
    input_kernel=7,
):
    """
    A ResNet-style 2D architecture.
    """
    layers = []
    layers.append(
        init_linear(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                input_kernel,
                stride=input_stride,
                padding=input_kernel // 2,
            )
        )
    )
    layers.append(nn.SELU())

    block_layers = []
    for block_idx in range(num_blocks):
        sublayers = []
        residual_in = hidden_channels * (2 ** max(0, block_idx - 1))
        residual_out = hidden_channels * (2**block_idx)
        # Residual block with downsampling
        sublayers.append(
            nn.Sequential(
                init_linear(
                    nn.Conv2d(
                        residual_in,
                        residual_out,
                        kernel_size=1,
                    )
                )
                if residual_in != residual_out
                else nn.Identity(),
                nn.BatchNorm2d(residual_out) if batch_norm else nn.Identity(),
                init_linear(
                    nn.Conv2d(
                        residual_out,
                        residual_out,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                nn.SELU(),
                nn.BatchNorm2d(residual_out) if batch_norm else nn.Identity(),
                init_linear(
                    nn.Conv2d(
                        residual_out,
                        residual_out,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                nn.SELU(),
            )
        )
        for _ in range(num_layers - 1):
            sublayers.append(
                nn.Sequential(
                    nn.BatchNorm2d(hidden_channels * (2**block_idx))
                    if batch_norm
                    else nn.Identity(),
                    init_linear(
                        nn.Conv2d(
                            hidden_channels * (2**block_idx),
                            hidden_channels * (2**block_idx),
                            kernel_size=3,
                            padding=1,
                        )
                    ),
                    nn.SELU(),
                    nn.BatchNorm2d(hidden_channels * (2**block_idx))
                    if batch_norm
                    else nn.Identity(),
                    init_linear(
                        nn.Conv2d(
                            hidden_channels * (2**block_idx),
                            hidden_channels * (2**block_idx),
                            kernel_size=3,
                            padding=1,
                        )
                    ),
                    nn.SELU(),
                )
            )
        block_layers.append(
            nn.Sequential(
                nn.Sequential(*sublayers),
                init_linear(
                    nn.Conv2d(
                        hidden_channels * (2**block_idx),
                        hidden_channels * (2**block_idx),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),
                nn.SELU(),
            )
        )
    layers.append(nn.Sequential(*block_layers))
    layers.append(
        nn.BatchNorm2d(hidden_channels * (2 ** (num_blocks - 1)))
        if batch_norm
        else nn.Identity()
    )
    layers.append(
        init_linear(
            nn.Conv2d(
                hidden_channels * (2 ** (num_blocks - 1)),
                out_channels,
                kernel_size=1,
            )
        )
    )
    if global_pooling:
        # Use adaptive average pooling directly, followed by a flatten to create a feature vector.
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
    return nn.Sequential(*layers)
