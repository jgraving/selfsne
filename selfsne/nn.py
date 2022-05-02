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

# Perceiver adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
# Under the following License:
# MIT License

# Copyright (c) 2021 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn, einsum
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from math import pi, log
from functools import wraps

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
        self.proj = init_selu(nn.Linear(in_channels, out_channels))
        self.module = module

    def forward(self, x):
        return self.proj(x) + self.module(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class PosEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        idx = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(idx)


def TokenMask(p=0.5):
    return nn.Sequential(
        Rearrange("batch tokens features -> batch tokens features ()"),
        nn.Dropout2d(p),
        Rearrange("batch tokens features () ->  batch tokens features"),
    )


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


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        num_queries,
        logits_dim=None,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        decoder_ff=False,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.queries = nn.Parameter(torch.randn(num_queries, queries_dim))

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=dim,
        )
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))

        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head),
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(),
                        get_cross_ff(),
                        get_latent_attn(),
                        get_latent_ff(),
                    ]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_logits = (
            nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        )

    def forward(self, data, mask=None):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, "n d -> b n d", b=b)

        # layers

        for cross_attn, cross_ff, self_attn, self_ff in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            x = self_attn(x) + x
            x = self_ff(x) + x

        # make sure queries contains batch dimension

        if self.queries.ndim == 2:
            queries = repeat(self.queries, "n d -> b n d", b=b)

        # cross attend from decoder queries to latents

        latents = self.decoder_cross_attn(queries, context=x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)
