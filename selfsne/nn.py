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
from torch import nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from selfsne.utils import (
    stop_gradient,
    random_sample_columns,
    straight_through_estimator,
)

import warnings


RSQRT2 = 2 ** -0.5


def lecun_normal_(x, mode="fan_in"):
    return init.kaiming_normal_(x, mode=mode, nonlinearity="linear")


def init_selu(x):
    lecun_normal_(x.weight)
    if hasattr(x, "bias"):
        if x.bias is not None:
            init.zeros_(x.bias)
    return x


class Residual(nn.Module):
    def __init__(self, module, residual=nn.Identity()):
        super().__init__()
        self.module = module
        self.residual = residual

    def forward(self, x):
        return (self.residual(x) + self.module(x)) * RSQRT2


def ParametricResidual(in_features, out_features, module):
    return Residual(
        module,
        init_selu(nn.Linear(in_features, out_features)),
    )


def Residual1d(in_features, out_features, module):
    return Residual(
        module,
        init_selu(nn.Linear(in_features, out_features))
        if in_features != out_features
        else nn.Identity(),
    )


def Residual2d(in_channels, out_channels, module):
    return Residual(
        module,
        init_selu(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        if in_channels != out_channels
        else nn.Identity(),
    )


class VarPool1d(nn.AvgPool1d):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        self.scale = kernel_size ** -0.5

    def forward(self, x):
        y = super().forward(x) * self.kernel_size
        return y * self.scale


class VarPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.scale = (self.kernel_size[0] * self.kernel_size[1]) ** -0.5

    def forward(self, x):
        y = super().forward(x) * self.kernel_size[0] * self.kernel_size[1]
        return y * self.scale


class VarPool3d(nn.AvgPool3d):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.scale = (
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        ) ** -0.5

    def forward(self, x):
        y = (
            super().forward(x)
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
        )
        return y * self.scale


class GlobalVarPool(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.sum(self.dim) * x.shape[self.dim] ** -0.5


class GlobalVarPool1d(nn.Module):
    def forward(self, x):
        y = reduce(x, "b c l -> b c", "sum") * x.shape[-1] ** -0.5
        return y


class GlobalVarPool2d(nn.Module):
    def forward(self, x):
        return reduce(x, "b c h w -> b c", "sum") * (x.shape[-1] * x.shape[-2]) ** -0.5


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class BatchCenter(nn.Module):
    def __init__(self, num_features, momentum=0.9):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros((1, num_features)))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0, keepdim=True)
            self.running_mean = torch.lerp(self.running_mean, batch_mean, self.momentum)
            return x - batch_mean
        else:
            return x - self.running_mean


class Bias(torch.nn.Module):
    def __init__(self, num_features):
        super(Bias, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x):
        return x + self.bias


class ProjectedBias(nn.Module):
    def __init__(self, num_features: int, param_dim: int):
        super(ProjectedBias, self).__init__()
        self.param = nn.Parameter(torch.randn(num_features, param_dim))
        self.projection = init_selu(nn.Linear(param_dim, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.projection(self.param).expand_as(input)
        return input + bias


class PairSampler(nn.Module):
    def __init__(self, x_sampler=nn.Identity(), y_sampler=nn.Identity()):
        super().__init__()
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler

    def forward(self, x):
        return self.x_sampler(x), self.y_sampler(x)


class PairAugmenter(nn.Module):
    def __init__(self, augmenter):
        super().__init__()
        self.augmenter = augmenter

    def forward(self, x):
        return torch.chunk(self.augmenter(torch.cat([x, x], dim=0)), 2, dim=0)


class ImageNetNorm(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        loc = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        scale = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", loc)

    def forward(self, x):
        return (x - self.loc) / self.scale


class StopGradient(nn.Module):
    def forward(self, x):
        return stop_gradient(x)


class InputNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features: int, eps: float = 1e-05, momentum=None) -> None:
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The batch normalization forward pass will be done differently based on the input dimension
        # Use super's forward to update running statistics
        super().forward(x)
        if x.dim() == 2:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        elif x.dim() == 3:
            # Reshape running mean and var to 3D
            running_mean = self.running_mean.view(1, -1, 1)
            running_var = self.running_var.view(1, -1, 1)
            # Use running statistics for normalization
            x_norm = (x - running_mean) / torch.sqrt(running_var + self.eps)
        else:
            raise ValueError(f"Unexpected input dimension {x.dim()}, expected 2 or 3.")
        return x_norm


class InputNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-05, momentum=None) -> None:
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use super's forward to update running statistics
        super().forward(x)
        # Use running statistics for normalization
        x_norm = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1, 1) + self.eps
        )
        return x_norm


class InputNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features: int, eps: float = 1e-05, momentum=None) -> None:
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use super's forward to update running statistics
        super().forward(x)
        # Use running statistics for normalization
        x_norm = (x - self.running_mean.view(1, -1, 1, 1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1, 1, 1) + self.eps
        )
        return x_norm


class PadShift(nn.Module):
    def forward(self, x):
        return F.pad(x, pad=(1, 0))[..., :-1]


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
    """Temporal Convolution Network (TCN)"""
    conv1d = CausalConv1d if causal else Conv1d
    return nn.Sequential(
        PadShift() if causal and causal_shift else nn.Identity(),
        init_selu(nn.Conv1d(in_channels, hidden_channels, 1)),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            *[
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
                                for dilation in 2 ** np.arange(num_layers)
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
    num_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    batch_norm=False,
):
    """Temporal Convolution Network (TCN)"""
    conv = nn.Conv2d
    return nn.Sequential(
        init_selu(nn.Conv2d(in_channels, hidden_channels, (1, 1))),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            *[
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
                                for dilation in 2 ** np.arange(num_layers)
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
    num_layers=4,
    n_blocks=4,
    causal=False,
    causal_shift=False,
    residual=False,
    batch_norm=False,
):
    """Temporal Convolution Network (TCN)"""
    conv = nn.Conv3d
    return nn.Sequential(
        init_selu(nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    Residual(
                        nn.Sequential(
                            *[
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
                                for dilation in 2 ** np.arange(num_layers)
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
    num_layers=2,
    num_blocks=4,
    global_pooling=True,
    batch_norm=False,
    input_stride=2,
    input_kernel=7,
):
    return nn.Sequential(
        init_selu(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                input_kernel,
                stride=input_stride,
                padding=input_kernel // 2,
            )
        ),
        nn.SELU(),
        nn.Sequential(
            *[
                nn.Sequential(
                    nn.Sequential(
                        Residual2d(
                            in_channels=hidden_channels
                            * (2 ** np.maximum(0, block_idx - 1)),
                            out_channels=hidden_channels * (2 ** block_idx),
                            module=nn.Sequential(
                                nn.BatchNorm2d(
                                    hidden_channels
                                    * (2 ** np.maximum(0, block_idx - 1))
                                )
                                if batch_norm
                                else nn.Identity(),
                                init_selu(
                                    nn.Conv2d(
                                        hidden_channels
                                        * (2 ** np.maximum(0, block_idx - 1)),
                                        hidden_channels * (2 ** block_idx),
                                        kernel_size=3,
                                        padding=1,
                                    )
                                ),
                                nn.SELU(),
                                nn.BatchNorm2d(hidden_channels * (2 ** block_idx))
                                if batch_norm
                                else nn.Identity(),
                                init_selu(
                                    nn.Conv2d(
                                        hidden_channels * (2 ** block_idx),
                                        hidden_channels * (2 ** block_idx),
                                        kernel_size=3,
                                        padding=1,
                                    )
                                ),
                                nn.SELU(),
                            ),
                        ),
                        *[
                            Residual2d(
                                in_channels=hidden_channels * (2 ** block_idx),
                                out_channels=hidden_channels * (2 ** block_idx),
                                module=nn.Sequential(
                                    nn.BatchNorm2d(hidden_channels * (2 ** block_idx))
                                    if batch_norm
                                    else nn.Identity(),
                                    init_selu(
                                        nn.Conv2d(
                                            hidden_channels * (2 ** block_idx),
                                            hidden_channels * (2 ** block_idx),
                                            kernel_size=3,
                                            padding=1,
                                        )
                                    ),
                                    nn.SELU(),
                                    nn.BatchNorm2d(hidden_channels * (2 ** block_idx))
                                    if batch_norm
                                    else nn.Identity(),
                                    init_selu(
                                        nn.Conv2d(
                                            hidden_channels * (2 ** block_idx),
                                            hidden_channels * (2 ** block_idx),
                                            kernel_size=3,
                                            padding=1,
                                        )
                                    ),
                                    nn.SELU(),
                                ),
                            )
                            for _ in range(num_layers - 1)
                        ],
                    ),
                    init_selu(
                        nn.Conv2d(
                            hidden_channels * (2 ** block_idx),
                            hidden_channels * (2 ** block_idx),
                            3,
                            stride=2,
                            padding=1,
                        )
                    ),
                    nn.SELU(),
                )
                for block_idx in range(num_blocks)
            ]
        ),
        nn.BatchNorm2d(hidden_channels * (2 ** (num_blocks - 1)))
        if batch_norm
        else nn.Identity(),
        init_selu(
            nn.Conv2d(hidden_channels * (2 ** (num_blocks - 1)), out_channels, 1)
        ),
        GlobalVarPool2d() if global_pooling else nn.Identity(),
    )


def MLP(
    in_features,
    out_features,
    hidden_features=256,
    num_layers=1,
    batch_norm=False,
):
    return nn.Sequential(
        init_selu(nn.Linear(in_features, hidden_features)),
        nn.SELU(),
        nn.Sequential(
            *[
                nn.Sequential(
                    nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity(),
                    init_selu(nn.Linear(hidden_features, hidden_features)),
                    nn.SELU(),
                )
                for _ in range(num_layers)
            ]
        ),
        nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity(),
        init_selu(nn.Linear(hidden_features, out_features)),
    )


def MultiheadMLP(
    in_features,
    out_features,
    hidden_features=256,
    num_layers=1,
    batch_norm=False,
    num_heads=1,
):
    return nn.Sequential(
        MultiheadLinear(in_features, hidden_features, num_heads),
        nn.SELU(),
        nn.Sequential(
            *[
                nn.Sequential(
                    nn.BatchNorm1d(hidden_features * num_heads)
                    if batch_norm
                    else nn.Identity(),
                    MultiheadLinear(hidden_features, hidden_features, num_heads),
                    nn.SELU(),
                )
                for _ in range(num_layers)
            ]
        ),
        nn.BatchNorm1d(hidden_features * num_heads) if batch_norm else nn.Identity(),
        MultiheadLinear(hidden_features, out_features, num_heads),
    )


class MultiheadLinear(nn.Module):
    def __init__(self, in_features, out_features, num_heads, bias=True):
        super(MultiheadLinear, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, num_heads))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.in_features ** -0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.dim() == 2:  # Input is 2D batch x features
            out = torch.einsum("bi,oih->bho", x, self.weight)
        elif x.dim() == 3:  # Input is 3D batch x num_heads x features
            if x.shape[1] != self.num_heads:
                raise ValueError(
                    f"Expected input shape[1] to be {self.num_heads} but got {x.shape[1]}"
                )
            out = torch.einsum("bhi,oih->bho", x, self.weight)

        else:
            raise ValueError(
                "Input must be 2D batch x features or 3D batch x num_heads x features"
            )

        if self.bias is not None:
            out += self.bias
        return out


def MultiheadEncoder(encoder, num_heads, hidden_dim, embedding_dim, num_layers=1):
    return nn.Sequential(
        encoder,
        MultiheadMLP(
            hidden_dim,
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ),
    )


class MultistageEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        num_stages,
        hidden_dim,
        embedding_dim,
        stage_hidden_layers=1,
        projector_hidden_layers=1,
    ):
        super().__init__()
        self.encoder = encoder
        self.stages = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=stage_hidden_layers)
                for _ in range(num_stages)
            ]
        )
        self.projector = MultiheadMLP(
            hidden_dim,
            embedding_dim,
            hidden_dim,
            num_layers=projector_hidden_layers,
            num_heads=num_stages,
        )

    def forward(self, x):
        h = self.encoder(x)
        stages = []
        for stage in self.stages:
            h = stage(h)
            stages.append(h)
        stages = torch.stack(stages, dim=1)
        return self.projector(stages)


class PositionEmbedding2d(nn.Module):
    def __init__(self, embedding_dim, height, width):
        super(PositionEmbedding2d, self).__init__()
        self.embedding_dim = embedding_dim
        self.height = height
        self.width = width

        # Initialize the height and width embeddings as learnable parameters
        self.height_embedding = nn.Parameter(torch.randn(embedding_dim, height))
        self.width_embedding = nn.Parameter(torch.randn(embedding_dim, width))

    def forward(self, x):
        # Create a tensor of shape (height, width, embedding_dim)
        # with embeddings for each height and width
        height_embed = self.height_embedding.view(1, self.embedding_dim, self.height, 1)
        width_embed = self.width_embedding.view(1, self.embedding_dim, 1, self.width)
        pos_embed = (height_embed + width_embed) * RSQRT2
        # Repeat the pos_embed for the entire batch
        pos_embed = pos_embed.repeat(x.shape[0], 1, 1, 1)
        # Add the position embedding 2D to the input tensor
        return torch.cat([x, pos_embed], dim=1)


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_positions):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_positions, embedding_dim))

    def forward(self, x):
        return (x + self.embedding[: x.shape[1]]) * RSQRT2


class TokenSampler(nn.Module):
    def __init__(self, p=None, num_tokens=None):
        super(TokenSampler, self).__init__()
        self.p = p
        self.num_tokens = num_tokens

        if self.p is None and self.num_tokens is None:
            warnings.warn(
                "Neither 'p' nor 'num_tokens' were specified. The module will not perform any sampling on the input tensor."
            )
        elif self.p is not None and (self.p <= 0 or self.p > 1):
            raise ValueError("The 'p' parameter must be between 0 and 1.")

    def forward(self, x):
        if self.training:
            batch_size, tokens, features = x.size()

            if self.num_tokens is not None:
                # Ensure that we don't sample more tokens than are available
                tokens_to_keep = min(tokens, self.num_tokens)
            elif self.p is not None:
                # Calculate the number of tokens to keep per batch
                tokens_to_keep = int(tokens * self.p)
            else:
                # If neither p nor num_tokens is specified, don't do anything
                return x

            # Use the random_sample_columns function to sample the remaining tokens for each item in the batch
            output_tensor = random_sample_columns(x, tokens_to_keep)

            return output_tensor
        else:
            return x


class PairedTokenSampler(nn.Module):
    def __init__(self, p=0.5, p_a=None, p_b=None):
        super(PairedTokenSampler, self).__init__()

        if p_a is None:
            p_a = p

        if p_b is None:
            p_b = 1 - p_a

        self.p_a = p_a
        self.p_b = p_b

        if p_a is not None and (p_a <= 0 or p_a > 1):
            raise ValueError("The 'p_a' parameter must be between 0 and 1.")
        if p_b is not None and (p_b <= 0 or p_b > 1):
            raise ValueError("The 'p_b' parameter must be between 0 and 1.")
        if p_a + p_b > 1:
            warnings.warn(
                "The sum of p_a and p_b is greater than 1, which may result in overlapping tokens."
            )

    def _calculate_samples(self, num_tokens):
        sample_a_tokens = int(num_tokens * self.p_a)
        sample_b_tokens = int(num_tokens * self.p_b)
        return sample_a_tokens, sample_b_tokens

    def forward(self, x):
        batch_size, num_tokens, num_features = x.shape

        # Calculate the number of tokens for each sample
        sample_a_tokens, sample_b_tokens = self._calculate_samples(num_tokens)

        # Generate random numbers for each token in the batch, ensuring they are on the same device
        rand_values = torch.randn_like(x[:, :, 0])

        # Use topk to get the indices of the highest k random values, where k is the number of tokens for each sample
        _, top_indices = torch.topk(rand_values, k=sample_a_tokens, dim=1)
        _, bottom_indices = torch.topk(
            rand_values, k=sample_b_tokens, dim=1, largest=False
        )

        # Use gather to sample the tokens into 'a' and 'b' using the indices
        a = x.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, num_features))
        b = x.gather(1, bottom_indices.unsqueeze(-1).expand(-1, -1, num_features))

        return a, b


class PairedImagePatchSampler(nn.Module):
    def __init__(self, patch_size, p=0.5, p_a=None, p_b=None):
        super(PairedImagePatchSampler, self).__init__()
        self.patch_size = patch_size
        self.token_sampler = PairedTokenSampler(p=p, p_a=p_a, p_b=p_b)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size

        # Ensure height and width are divisible by patch_size
        assert (
            height % patch_size == 0 and width % patch_size == 0
        ), "Height and width must be divisible by patch_size."

        # Rearrange the input image into patches
        patches = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )

        # Sample the patches using PairedTokenSampler
        sampled_patches_a, sampled_patches_b = self.token_sampler(patches)

        # Rearrange the sampled patches back into images
        a = rearrange(
            sampled_patches_a,
            "b (s) (p1 p2 c) -> b c (s p1) (p2)",
            p1=patch_size,
            p2=patch_size,
        )
        b = rearrange(
            sampled_patches_b,
            "b (s) (p1 p2 c) -> b c (s p1) (p2)",
            p1=patch_size,
            p2=patch_size,
        )

        return a, b


class PairedSequencePatchSampler(nn.Module):
    def __init__(self, patch_size, p=0.5, p_a=None, p_b=None):
        super(PairedSequencePatchSampler, self).__init__()
        self.patch_size = patch_size
        self.token_sampler = PairedTokenSampler(p=p, p_a=p_a, p_b=p_b)

    def forward(self, x):
        batch_size, channels, seq_length = x.shape
        patch_size = self.patch_size

        # Ensure seq_length is divisible by patch_size
        assert (
            seq_length % patch_size == 0
        ), "Seq_length must be divisible by patch_size."

        # Rearrange the input sequence into patches
        patches = rearrange(x, "b c (s p) -> b (s) (p c)", p=patch_size)

        # Sample the patches using PairedTokenSampler
        sampled_patches_a, sampled_patches_b = self.token_sampler(patches)

        # Rearrange the sampled patches back into sequences
        a = rearrange(sampled_patches_a, "b (s) (p c) -> b c (s p)", p=patch_size)
        b = rearrange(sampled_patches_b, "b (s) (p c) -> b c (s p)", p=patch_size)

        return a, b


class PairedCausalSequencePatchSampler(nn.Module):
    def __init__(self, patch_size, p=None, split_index=None):
        super(PairedCausalSequencePatchSampler, self).__init__()
        self.patch_size = patch_size
        self.p = p
        self.split_index = split_index

    def forward(self, x):
        batch_size, channels, seq_length = x.shape
        patch_size = self.patch_size

        # Ensure seq_length is divisible by patch_size
        assert (
            seq_length % patch_size == 0
        ), "Seq_length must be divisible by patch_size."

        # Rearrange the input sequence into patches
        patches = rearrange(x, "b c (s p) -> b (s) (p c)", p=patch_size)

        # Determine the split index
        if self.split_index is None and self.p is None:
            split_index = patches.size(1) - 1
        elif self.split_index is not None:
            split_index = min(patches.size(1) - 1, self.split_index)
        else:
            split_index = int(patches.size(1) * self.p)

        # Split the patches into a and b
        a = patches[:, :split_index, :]
        b = patches[:, split_index:, :]

        # Rearrange a and b back into sequences
        a = rearrange(a, "b (s) (p c) -> b c (s p)", p=patch_size)
        b = rearrange(b, "b (s) (p c) -> b c (s p)", p=patch_size)

        return a, b


def PatchEmbedding1d(seq_length, patch_size, embedding_dim, in_channels):
    assert (
        seq_length % patch_size
    ) == 0, "Sequence length must be divisible by patch size."
    return nn.Sequential(
        init_selu(
            nn.Conv1d(
                in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        ),
        Rearrange("b c s -> b s c"),
    )


def PatchEmbedding2d(image_size, patch_size, embedding_dim, in_channels=3):
    assert (image_size % patch_size) == 0, "Image size must be divisible by patch size."
    return nn.Sequential(
        init_selu(
            nn.Conv2d(
                in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        ),
        Rearrange("b c h w -> b (h w) c"),
    )


def PreNorm(in_features, module):
    return Residual(nn.Sequential(nn.LayerNorm(in_features), module))


def PostNorm(out_features, module):
    return nn.Sequential(Residual(module), nn.LayerNorm(out_features))


def ResNorm(out_features, module):
    return Residual(nn.Sequential(module, nn.LayerNorm(out_features)))


class VarSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        attention_dim=None,
        qk_dim=None,
        v_dim=None,
        output_dim=None,
        dropout=0.0,
        causal_mask=False,
    ):
        super(VarSelfAttention, self).__init__()
        self.input_dim = input_dim
        if attention_dim is None:
            assert (
                input_dim % num_heads == 0
            ), f"input_dim ({input_dim}) is not divisible by num_heads ({num_heads})"
            self.attention_dim = input_dim // num_heads
        else:
            self.attention_dim = attention_dim
        self.qk_dim = qk_dim if qk_dim is not None else self.attention_dim
        self.head_qk = self.qk_dim
        self.qk_dim *= num_heads
        self.v_dim = v_dim if v_dim is not None else self.attention_dim
        self.head_v = self.v_dim
        self.v_dim *= num_heads
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.scale = self.head_qk ** -0.5
        self.qk = init_selu(nn.Linear(self.input_dim, 2 * self.qk_dim))
        self.v = (
            init_selu(nn.Linear(self.input_dim, self.v_dim))
            if self.input_dim != self.v_dim
            else nn.Identity()
        )
        self.out = (
            init_selu(nn.Linear(self.v_dim, self.output_dim))
            if self.output_dim != self.v_dim
            else nn.Identity()
        )

        self.dropout = nn.AlphaDropout(p=dropout) if dropout > 0 else nn.Identity()
        self.causal_mask = causal_mask

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        qk = self.qk(x)  # shape (batch_size, seq_len, 2*qk_dim)
        qk = rearrange(
            qk, "b s (h d c) -> b h s d c", h=self.num_heads, d=self.head_qk, c=2
        )  # shape (batch_size, num_heads, seq_len, head_qk, 2)
        q, k = qk[..., 0], qk[..., 1]

        v = self.v(x)  # shape (batch_size, seq_len, v_dim)
        v = rearrange(
            v, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_v
        )  # shape (batch_size, num_heads, seq_len, head_v)

        # compute similarity
        # q shape: (batch_size, num_heads, seq_len_i, head_qk)
        # k shape: (batch_size, num_heads, seq_len_j, head_qk)
        # similarity shape: (batch_size, num_heads, seq_len_i, seq_len_j)
        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # scale dot product similarity
        similarity *= self.scale

        # apply activation and dropout
        attention = F.selu(similarity - torch.mean(similarity, dim=-1, keepdim=True))
        attention = self.dropout(attention)

        # apply causal mask if specified
        if self.causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attention.device), diagonal=1
            )
            mask = mask.view(1, 1, seq_len, seq_len).repeat(
                batch_size, self.num_heads, 1, 1
            )
            gradient = attention.masked_fill(
                mask == 0,
                -1.6732632423543772848170429916717,  # set masked elements to selu lower bound for zero grad
            )
            estimator = attention * mask
            attention = straight_through_estimator(
                gradient, estimator
            )  # use straight-through estimator to pass zeros in forward and backward

        # compute output
        # v shape: (batch_size, num_heads, seq_len_j, head_v)
        # out shape: (batch_size, num_heads, seq_len_i, head_v)
        out = (
            torch.einsum("b h i j, b h j d -> b h i d", attention, v)
            * (seq_len) ** -0.5
        )
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        attention_dim=None,
        qk_dim=None,
        v_dim=None,
        output_dim=None,
        dropout=0.0,
        causal_mask=False,
    ):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        if attention_dim is None:
            assert (
                input_dim % num_heads == 0
            ), f"input_dim ({input_dim}) is not divisible by num_heads ({num_heads})"
            self.attention_dim = input_dim // num_heads
        else:
            self.attention_dim = attention_dim
        self.qk_dim = qk_dim if qk_dim is not None else self.attention_dim
        self.head_qk = self.qk_dim
        self.qk_dim *= num_heads
        self.v_dim = v_dim if v_dim is not None else self.attention_dim
        self.head_v = self.v_dim
        self.v_dim *= num_heads
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.scale = self.head_qk ** -0.5
        self.qk = init_selu(nn.Linear(self.input_dim, 2 * self.qk_dim))
        self.v = (
            init_selu(nn.Linear(self.input_dim, self.v_dim))
            if self.input_dim != self.v_dim
            else nn.Identity()
        )
        self.out = (
            init_selu(nn.Linear(self.v_dim, self.output_dim))
            if self.output_dim != self.v_dim
            else nn.Identity()
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.causal_mask = causal_mask

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        qk = self.qk(x)  # shape (batch_size, seq_len, 2*qk_dim)
        qk = rearrange(
            qk, "b s (h d c) -> b h s d c", h=self.num_heads, d=self.head_qk, c=2
        )  # shape (batch_size, num_heads, seq_len, head_qk, 2)
        q, k = qk[..., 0], qk[..., 1]

        v = self.v(x)  # shape (batch_size, seq_len, v_dim)
        v = rearrange(
            v, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_v
        )  # shape (batch_size, num_heads, seq_len, head_v)

        # compute similarity
        # q shape: (batch_size, num_heads, seq_len_i, head_qk)
        # k shape: (batch_size, num_heads, seq_len_j, head_qk)
        # similarity shape: (batch_size, num_heads, seq_len_i, seq_len_j)
        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # apply causal mask if specified
        if self.causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=similarity.device), diagonal=1
            )
            mask = mask.view(1, 1, seq_len, seq_len).repeat(
                batch_size, self.num_heads, 1, 1
            )
            similarity = similarity.masked_fill(
                mask == 0, torch.finfo(similarity.dtype).min
            )  # float('-inf')

        # scale dot product similarity
        similarity *= self.scale

        # apply softmax and dropout
        attention = similarity.softmax(-1)
        attention = self.dropout(attention)

        # compute output
        # v shape: (batch_size, num_heads, seq_len_j, head_v)
        # out shape: (batch_size, num_heads, seq_len_i, head_v)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        num_latent_tokens,
        latent_dim=None,
        attention_dim=None,
        qk_dim=None,
        v_dim=None,
        output_dim=None,
        dropout=0.0,
        latent_prenorm=False,
    ):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = self.input_dim if latent_dim is None else latent_dim
        if attention_dim is None:
            assert (
                input_dim % num_heads == 0
            ), f"input_dim ({input_dim}) is not divisible by num_heads ({num_heads})"
            self.attention_dim = self.input_dim // num_heads
        else:
            self.attention_dim = attention_dim
        self.qk_dim = self.attention_dim if qk_dim is None else qk_dim
        self.head_qk = self.qk_dim
        self.qk_dim *= num_heads
        self.v_dim = v_dim if v_dim is not None else self.attention_dim
        self.head_v = self.v_dim
        self.v_dim *= num_heads
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.scale = self.head_qk ** -0.5
        self.k = init_selu(nn.Linear(self.input_dim, self.qk_dim))
        self.v = (
            init_selu(nn.Linear(self.input_dim, self.v_dim))
            if self.input_dim != self.v_dim
            else nn.Identity()
        )
        self.out = (
            init_selu(nn.Linear(self.v_dim, self.output_dim))
            if self.output_dim != self.v_dim
            else nn.Identity()
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Sequential(
            nn.Identity()
            if self.latent_dim == self.output_dim
            else init_selu(nn.Linear(self.latent_dim, self.output_dim)),
        )

        self.latents = nn.Parameter(
            torch.randn((1, num_latent_tokens, self.latent_dim))
        )
        self.q = nn.Sequential(
            nn.LayerNorm(self.latent_dim) if latent_prenorm else nn.Identity(),
            init_selu(nn.Linear(self.latent_dim, self.qk_dim)),
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        q = self.q(self.latents)  # shape (1, num_latent_tokens, qk_dim)
        q = rearrange(
            q, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_qk
        )  # shape (1, num_heads, num_latent_tokens, head_qk)
        q = q.repeat(
            batch_size, 1, 1, 1
        )  # shape (batch_size, num_heads, num_latent_tokens, head_qk)

        k = self.k(x)  # shape (batch_size, seq_len, qk_dim)
        k = rearrange(
            k, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_qk
        )  # shape (batch_size, num_heads, seq_len, head_qk)

        v = self.v(x)  # shape (batch_size, seq_len, v_dim)
        v = rearrange(
            v, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_v
        )  # shape (batch_size, num_heads, seq_len, head_v)

        # compute dot product similarity
        similarity = torch.einsum(
            "b h i d, b h j d -> b h i j", q, k
        )  # shape (batch_size, num_heads, num_latent_tokens, seq_len)

        # scale dot product similarity
        similarity *= self.scale

        # apply softmax and dropout
        attention = similarity.softmax(-1)
        attention = self.dropout(attention)

        # compute output
        # v shape: (batch_size, num_heads, seq_len, head_v)
        # out shape: (batch_size, num_heads, num_latent_tokens, head_v)
        out = torch.einsum(
            "b h i j, b h j d -> b h i d", attention, v
        )  # shape (batch_size, num_heads, num_latent_tokens, head_v)
        out = rearrange(
            out, "b h l d -> b l (h d)"
        )  # shape (batch_size, num_latent_tokens, v_dim)
        return (self.out(out) + self.residual(self.latents)) * RSQRT2


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_out,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        feedforward_multiplier=2,
    ):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim * num_heads,
                nhead=num_heads,
                dim_feedforward=embedding_dim * feedforward_multiplier,
                dropout=dropout,
                activation="gelu",
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embedding_dim, num_out)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
