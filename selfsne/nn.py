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

from selfsne.utils import stop_gradient, random_sample_columns

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
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return (x + self.module(x)) * RSQRT2


class PadShift(nn.Module):
    def forward(self, x):
        return F.pad(x, pad=(1, 0))[..., :-1]


class ParametricResidual(nn.Module):
    def __init__(self, in_features, out_features, module):
        super().__init__()
        self.proj = init_selu(nn.Linear(in_features, out_features))
        self.module = module

    def forward(self, x):
        return (self.proj(x) + self.module(x)) * RSQRT2


class VarPool2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.scale = (self.kernel_size[0] * self.kernel_size[1]) ** -0.5

    def forward(self, x):
        y = rearrange(
            x,
            "b c (h kh) (w kw) -> b c h w kh kw",
            kh=self.kernel_size[0],
            kw=self.kernel_size[1],
        )
        return reduce(y, "b c h w kh kw -> b c h w", "sum") * self.scale


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
        return reduce(x, "b c h w -> b c", "sum") * (x.shape[-1] * x.shape[2]) ** -0.5


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
    def __init__(self, num_features: int, eps: float = 1e-05) -> None:
        super().__init__(num_features, eps=eps, momentum=None, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use super's forward to update running statistics
        super().forward(x)
        # Use running statistics for normalization
        x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return x_norm


class InputNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-05) -> None:
        super().__init__(num_features, eps=eps, momentum=None, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use super's forward to update running statistics
        super().forward(x)
        # Use running statistics for normalization
        x_norm = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1, 1) + self.eps
        )
        return x_norm


class InputNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features: int, eps: float = 1e-05) -> None:
        super().__init__(num_features, eps=eps, momentum=None, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use super's forward to update running statistics
        super().forward(x)
        # Use running statistics for normalization
        x_norm = (x - self.running_mean.view(1, -1, 1, 1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1, 1, 1) + self.eps
        )
        return x_norm


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
    n_layers=4,
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
                    nn.BatchNorm2d(hidden_channels * (2 ** block_idx))
                    if batch_norm
                    else nn.Identity(),
                    init_selu(
                        nn.Conv2d(
                            hidden_channels * (2 ** block_idx),
                            hidden_channels * (2 ** (block_idx + 1)),
                            1,
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            *[
                                nn.Sequential(
                                    nn.BatchNorm2d(
                                        hidden_channels * (2 ** (block_idx + 1))
                                    )
                                    if batch_norm
                                    else nn.Identity(),
                                    init_selu(
                                        nn.Conv2d(
                                            hidden_channels * (2 ** (block_idx + 1)),
                                            hidden_channels * (2 ** (block_idx + 1)),
                                            kernel_size=3,
                                            padding=1,
                                        )
                                    ),
                                    nn.SELU(),
                                )
                                for _ in range(n_layers)
                            ]
                        )
                    ),
                    VarPool2d(2),
                )
                for block_idx in range(n_blocks)
            ]
        ),
        nn.BatchNorm2d(hidden_channels * (2 ** n_blocks))
        if batch_norm
        else nn.Identity(),
        init_selu(nn.Conv2d(hidden_channels * (2 ** n_blocks), out_channels, 1)),
        GlobalVarPool2d() if global_pooling else nn.Identity(),
    )


def MLP(
    in_features,
    out_features,
    hidden_features=256,
    n_layers=4,
    batch_norm=False,
):
    net = nn.Sequential(
        init_selu(nn.Linear(in_features, hidden_features)),
        nn.SELU(),
        Residual(
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.BatchNorm1d(hidden_features)
                        if batch_norm
                        else nn.Identity(),
                        init_selu(nn.Linear(hidden_features, hidden_features)),
                        nn.SELU(),
                    )
                    for _ in range(n_layers)
                ]
            )
        ),
        nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity(),
        init_selu(nn.Linear(hidden_features, out_features)),
    )
    return (
        Residual(net)
        if in_features == out_features
        else ParametricResidual(in_features, out_features, net)
    )


class PositionEmbedding2d(nn.Module):
    def __init__(self, embedding_dim, height, width):
        super(PositionEmbedding2D, self).__init__()
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
        # Add the position embedding 2D to the input tensor
        return (x + pos_embed) * RSQRT2


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_positions):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_positions, embedding_dim))

    def forward(self, x):
        return (x + self.embedding) * RSQRT2


class SampleTokens(nn.Module):
    def __init__(self, p=0.5):
        super(SampleTokens, self).__init__()
        self.p = p

    def forward(self, tensor):
        if self.training:
            batch_size, tokens, features = tensor.size()

            # Calculate the number of tokens to keep per batch
            tokens_to_keep = int(tokens * self.p)

            # Use the random_sample_columns function to sample the remaining tokens for each item in the batch
            output_tensor = random_sample_columns(tensor, tokens_to_keep)

            return output_tensor
        else:
            return tensor


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, input_channels=3):
        super().__init__()
        assert (
            image_size % patch_size
        ) == 0, "Image size must be divisible by patch size."
        self.patch_size = patch_size
        self.patch_embedding = init_selu(
            nn.Conv2d(
                input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        return rearrange(x, "b c h w -> b (h w) c")


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        attention_dim,
        num_heads,
        qk_dim=None,
        v_dim=None,
        output_dim=None,
        dropout=0.1,
        causal_mask=False,
        prenorm=True,
    ):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.qk_dim = qk_dim if qk_dim is not None else attention_dim
        self.head_qk = self.qk_dim
        self.qk_dim *= num_heads
        self.v_dim = v_dim if v_dim is not None else attention_dim
        self.head_v = self.v_dim
        self.v_dim *= num_heads
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.scale = self.head_qk ** -0.5
        self.qk = init_selu(nn.Linear(input_dim, 2 * self.qk_dim))
        self.v = (
            init_selu(nn.Linear(input_dim, self.v_dim))
            if input_dim != self.v_dim
            else nn.Identity()
        )
        self.out = (
            init_selu(nn.Linear(self.v_dim, self.output_dim))
            if self.output_dim != self.v_dim
            else nn.Identity()
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if input_dim == self.output_dim
            else init_selu(nn.Linear(input_dim, self.output_dim))
        )
        self.causal_mask = causal_mask
        self.prenorm = nn.LayerNorm(self.output_dim) if prenorm else nn.Identity()

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
        return (self.prenorm(self.out(out)) + self.residual(x)) * RSQRT2


class CrossAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_latent_tokens,
        attention_dim,
        num_heads,
        qk_dim=None,
        v_dim=None,
        output_dim=None,
        dropout=0.1,
        prenorm=True,
    ):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.qk_dim = qk_dim if qk_dim is not None else attention_dim
        self.head_qk = self.qk_dim
        self.qk_dim *= num_heads
        self.v_dim = v_dim if v_dim is not None else attention_dim
        self.head_v = self.v_dim
        self.v_dim *= num_heads
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.scale = self.head_qk ** -0.5
        self.k = init_selu(nn.Linear(input_dim, self.qk_dim))
        self.v = init_selu(nn.Linear(input_dim, self.v_dim))
        self.out = init_selu(nn.Linear(self.v_dim, self.output_dim))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if self.latent_dim == self.output_dim
            else init_selu(nn.Linear(self.latent_dim, self.output_dim))
        )

        self.latents = nn.Parameter(torch.randn((1, num_latent_tokens, latent_dim)))
        self.q = init_selu(nn.Linear(latent_dim, self.qk_dim))
        self.prenorm = nn.LayerNorm(self.output_dim) if prenorm else nn.Identity()

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
        return (self.prenorm(self.out(out)) + self.residual(self.latents)) * RSQRT2


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
