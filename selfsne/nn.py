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

from einops import rearrange
from einops.layers.torch import Rearrange

from selfsne.utils import (
    stop_gradient,
    random_sample_columns,
    straight_through_estimator,
)

import warnings


def lecun_normal_(x, mode="fan_in"):
    return init.kaiming_normal_(x, mode=mode, nonlinearity="linear")


def init_linear(x):
    lecun_normal_(x.weight)
    if hasattr(x, "bias"):
        if x.bias is not None:
            init.zeros_(x.bias)
    return x


init_selu = init_linear


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
    return Residual(
        module,
        init_linear(nn.Linear(in_features, out_features))
        if in_features != out_features
        else nn.Identity(),
    )


def Residual2d(in_channels, out_channels, module):
    return Residual(
        module,
        init_linear(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        if in_channels != out_channels
        else nn.Identity(),
    )


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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
        init_linear(nn.Conv1d(in_channels, hidden_channels, 1)),
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
                                for dilation in 2 ** np.arange(num_layers)
                            ]
                        )
                    )
                    for _ in range(n_blocks)
                ]
            )
        ),
        nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity(),
        init_linear(nn.Conv1d(hidden_channels, out_channels, 1)),
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
        init_linear(nn.Conv2d(in_channels, hidden_channels, (1, 1))),
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
                                    init_linear(
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
        init_linear(nn.Conv2d(hidden_channels, out_channels, (1, 1))),
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
        init_linear(nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))),
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
                                    init_linear(
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
        init_linear(nn.Conv3d(hidden_channels, out_channels, (1, 1, 1))),
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
        init_linear(
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
                                init_linear(
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
                                init_linear(
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
                                    init_linear(
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
                                    init_linear(
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
                    init_linear(
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
        init_linear(
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
        init_linear(nn.Linear(in_features, hidden_features)),
        nn.SELU(),
        nn.Sequential(
            *[
                nn.Sequential(
                    nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity(),
                    init_linear(nn.Linear(hidden_features, hidden_features)),
                    nn.SELU(),
                )
                for _ in range(num_layers)
            ]
        ),
        nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity(),
        init_linear(nn.Linear(hidden_features, out_features)),
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


class Multilinear(nn.Module):
    def __init__(self, embedding_dim, order):
        super(Multilinear, self).__init__()
        self.embedding_dim = embedding_dim
        self.order = max(1, order)  # Ensure order is at least 1
        self.num_chunks = self.order + 1  # e.g., order=1 implies 2 chunks

        self.linear = init_linear(
            nn.Linear(embedding_dim, self.num_chunks * embedding_dim, bias=True)
        )

    def forward(self, x):
        # x is expected to be of shape (..., embedding_dim)
        orig_shape = x.shape  # save original shape
        # Flatten all dimensions except the last (feature) dimension
        x_flat = x.view(-1, self.embedding_dim)

        # Apply the linear transformation: shape becomes (-1, num_chunks * embedding_dim)
        transformed = self.linear(x_flat)

        # Reshape to separate chunks: (-1, num_chunks, embedding_dim)
        transformed = transformed.view(-1, self.num_chunks, self.embedding_dim)

        # Compute interactions: start with elementwise multiplication of first two chunks
        interaction = transformed[:, 0] * transformed[:, 1]
        out_flat = interaction.clone()

        # Multiply in additional chunks cumulatively and sum their contribution
        for idx in range(2, self.num_chunks):
            interaction = interaction * transformed[:, idx]
            out_flat = out_flat + interaction

        # Residual connection: add the computed interactions back to the original input
        output = x_flat + out_flat

        # Restore original shape and return
        output = output.view(*orig_shape)
        return output


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
        pos_embed = height_embed + width_embed
        # Repeat the pos_embed for the entire batch
        pos_embed = pos_embed.repeat(x.shape[0], 1, 1, 1)
        # Add the position embedding 2D to the input tensor
        return torch.cat([x, pos_embed], dim=1)


class PositionEmbedding1d(nn.Module):
    def __init__(self, embedding_dim, num_positions):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_positions, embedding_dim))

    def forward(self, x):
        return x + self.embedding[: x.shape[1]]


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


def PatchEmbedding1d(seq_length, patch_size, embedding_dim, in_channels):
    assert (
        seq_length % patch_size
    ) == 0, "Sequence length must be divisible by patch size."
    return nn.Sequential(
        init_linear(
            nn.Conv1d(
                in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        ),
        Rearrange("b c s -> b s c"),
    )


def PatchEmbedding2d(image_size, patch_size, embedding_dim, in_channels=3):
    assert (image_size % patch_size) == 0, "Image size must be divisible by patch size."
    return nn.Sequential(
        init_linear(
            nn.Conv2d(
                in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        ),
        Rearrange("b c h w -> b (h w) c"),
    )


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Computes and applies Sinusoidal Positional Embeddings to the input tokens.
    Assumes input tensor shape (batch_size, seq_len, model_dim).
    """

    def __init__(self, model_dim: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.model_dim = model_dim
        # Precompute the scaling factors (div_term) as a buffer
        self.register_buffer(
            "div_term",
            torch.exp(
                torch.arange(0, model_dim, 2)
                * -(torch.log(torch.tensor(10000.0)) / model_dim)
            ),
            persistent=False,  # Not saved with model state_dict
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds sinusoidal positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Tensor with sinusoidal positional embeddings added, same shape as input.
        """
        batch_size, seq_len, model_dim = x.shape
        assert (
            model_dim == self.model_dim
        ), "Input model_dim must match initialized model_dim."

        # Compute positions dynamically
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(
            1
        )  # (seq_len, 1)

        # Use precomputed div_term directly
        sin_emb = torch.sin(position * self.div_term)  # (seq_len, model_dim / 2)
        cos_emb = torch.cos(position * self.div_term)  # (seq_len, model_dim / 2)

        # Rearrange and concatenate sine and cosine embeddings
        positional_embedding = rearrange(
            [sin_emb, cos_emb], "d seq half_dim -> seq (d half_dim)"
        )  # (seq_len, model_dim)

        # Add positional embeddings to input tensor
        positional_embedding = rearrange(
            positional_embedding, "seq dim -> 1 seq dim"
        ).expand(
            batch_size, -1, -1
        )  # (batch_size, seq_len, model_dim)
        return x + positional_embedding


class RotaryPositionalEmbedding(nn.Module):
    """
    Computes and applies Rotary Positional Embeddings (RoPE) to the input tokens.
    Assumes input tensor shape (batch_size, seq_len, model_dim).
    """

    def __init__(self, model_dim: int):
        super(RotaryPositionalEmbedding, self).__init__()
        self.model_dim = model_dim
        # Precompute the scaling factors (div_term) as a buffer
        self.register_buffer(
            "div_term",
            torch.exp(
                torch.arange(0, model_dim, 2)
                * -(torch.log(torch.tensor(10000.0)) / model_dim)
            ),
            persistent=False,  # Not saved with model state_dict
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        batch_size, seq_len, model_dim = x.shape
        assert (
            model_dim == self.model_dim
        ), "Input model_dim must match initialized model_dim."

        # Compute positions dynamically
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(
            1
        )  # (seq_len, 1)

        # Use precomputed div_term directly
        sin_emb = rearrange(
            torch.sin(position * self.div_term), "seq half_dim -> 1 seq half_dim"
        )  # (1, seq_len, model_dim / 2)
        cos_emb = rearrange(
            torch.cos(position * self.div_term), "seq half_dim -> 1 seq half_dim"
        )  # (1, seq_len, model_dim / 2)

        # Split x into even and odd indices for rotary application
        x_even = rearrange(x[..., 0::2], "b seq half_dim -> b seq half_dim")
        x_odd = rearrange(x[..., 1::2], "b seq half_dim -> b seq half_dim")

        # Apply RoPE
        x1 = x_even * cos_emb - x_odd * sin_emb  # (batch_size, seq_len, model_dim / 2)
        x2 = x_even * sin_emb + x_odd * cos_emb  # (batch_size, seq_len, model_dim / 2)

        # Concatenate even and odd back together
        return rearrange(
            [x1, x2], "d b seq half_dim -> b seq (d half_dim)"
        )  # (batch_size, seq_len, model_dim)


class GlobalTokens(nn.Module):
    def __init__(self, num_global_tokens, model_dim):
        """
        Args:
            num_global_tokens (int): Number of global tokens to add.
            model_dim (int): Dimension of the model (dimensionality of the tokens).
        """
        super(GlobalTokens, self).__init__()
        self.num_global_tokens = num_global_tokens
        self.model_dim = model_dim
        # Define global tokens as trainable parameters
        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, model_dim))

    def forward(self, x):
        """
        Concatenate global tokens to the input tokens.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, model_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens + num_global_tokens, model_dim).
        """
        batch_size = x.size(0)
        # Repeat global tokens for the entire batch
        global_tokens_batch = self.global_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, num_global_tokens, model_dim)
        # Concatenate global tokens to the input tokens
        return torch.cat(
            [x, global_tokens_batch], dim=1
        )  # (batch_size, num_tokens + num_global_tokens, model_dim)


class GatedMLP(nn.Module):
    def __init__(self, model_dim, expansion_factor=4, activation=None):
        super().__init__()
        # Use SELU as the default activation, but allow a custom activation if provided.
        self.activation = activation if activation is not None else nn.SELU()

        # Compute the expanded hidden dimension.
        hidden_dim = model_dim * expansion_factor

        # The projection layer expands the model_dim to 2 * hidden_dim.
        self.proj = init_linear(nn.Linear(model_dim, hidden_dim * 2))

        # The output layer compresses the gated representation back to model_dim.
        self.out = init_linear(nn.Linear(hidden_dim, model_dim))

    def forward(self, x):
        # x shape: (batch_size, ..., model_dim)
        x_proj = self.proj(x)  # Shape: (batch_size, ..., 2 * hidden_dim)
        # Split into two equal parts along the last dimension.
        x_a, x_b = x_proj.chunk(2, dim=-1)
        # Apply the activation to one half and multiply element-wise with the other.
        gated = x_a * self.activation(x_b)
        # Compress back to the original model_dim.
        return self.out(gated)


def PreLayerNorm(in_features, module):
    return Residual(nn.Sequential(nn.LayerNorm(in_features), module))


def PreRMSNorm(in_features, module):
    return Residual(nn.Sequential(nn.RMSNorm(in_features), module))


class Attention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0, causal_mask=False):
        super(Attention, self).__init__()
        assert (
            model_dim % num_heads == 0
        ), f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.causal_mask = causal_mask

    def forward(self, q, k, v, mask=None):
        # Compute scaled dot-product attention
        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # Apply causal mask if specified
        if self.causal_mask:
            seq_len = similarity.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=similarity.device), diagonal=1
            ).bool()
            similarity.masked_fill_(causal_mask, float("-inf"))

        # Apply custom mask if provided
        if mask is not None:
            similarity.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attention = similarity.softmax(dim=-1)
        attention = self.dropout(attention)

        # Compute attention output
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        return out


class SelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0, causal_mask=False):
        super(SelfAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.qkv = init_linear(nn.Linear(model_dim, model_dim * 3))
        self.out = init_linear(nn.Linear(model_dim, model_dim))
        self.attention = Attention(model_dim, num_heads, dropout, causal_mask)

    def forward(self, x):

        # Generate Q, K, V from input
        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, "b s (h d c) -> c b h s d", h=self.num_heads, d=self.head_dim, c=3
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply attention
        out = self.attention(q, k, v)

        # Rearrange and apply output linear layer
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear layers for Q, K, V
        self.q = init_linear(nn.Linear(model_dim, model_dim))
        self.kv = init_linear(nn.Linear(model_dim, model_dim * 2))
        self.out = init_linear(nn.Linear(model_dim, model_dim))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Attention module
        self.attention = Attention(model_dim, num_heads, dropout)

    def forward(self, query, context, mask=None):
        """
        Args:
            query: Tensor of shape [batch_size, query_len, model_dim]
            context: Tensor of shape [batch_size, context_len, model_dim]
            mask: Optional mask for attention computation
        """
        batch_size, query_len, _ = query.shape

        # Generate Q from query and K, V from context
        q = self.q(query)
        kv = self.kv(context)
        kv = rearrange(
            kv, "b s (h d c) -> c b h s d", h=self.num_heads, d=self.head_dim, c=2
        )
        k, v = kv[0], kv[1]

        # Rearrange Q for multi-head
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)

        # Apply attention
        out = self.attention(q, k, v, mask=mask)

        # Rearrange and apply output linear layer
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class LatentCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads, num_latent_tokens, dropout=0.0):
        super(LatentCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim

        # Learnable latent tokens
        self.latents = nn.Parameter(
            torch.randn(1, num_latent_tokens, model_dim) / np.sqrt(model_dim)
        )

        # Generic cross-attention
        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, context_len, model_dim]
            mask: Optional mask for attention computation
        """
        batch_size, _, _ = x.shape

        # Expand latent tokens for the current batch
        latents = self.latents.expand(batch_size, -1, -1)

        # Use generic cross-attention with latent queries
        return self.cross_attention(latents, x, mask=mask) + latents
