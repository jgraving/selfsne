# nn/linear.py
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
import numpy as np

from selfsne.nn.init import init_linear

__all__ = [
    "MLP",
    "MultiheadMLP",
    "MultiheadLinear",
    "MultiheadEncoder",
    "Multilinear",
    "GatedMLP",
]


def MLP(in_features, out_features, hidden_features=256, num_layers=1, batch_norm=False):
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
        if x.dim() == 2:  # Input is (batch_size, features)
            out = torch.einsum("bi,oih->bho", x, self.weight)
        elif x.dim() == 3:  # Input is (batch_size, num_heads, features)
            if x.shape[1] != self.num_heads:
                raise ValueError(
                    f"Expected input with num_heads={self.num_heads}, but got {x.shape[1]}"
                )
            out = torch.einsum("bhi,oih->bho", x, self.weight)
        else:
            raise ValueError(
                "Input must be 2D (batch, features) or 3D (batch, num_heads, features)"
            )
        if self.bias is not None:
            out += self.bias
        return out


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
        orig_shape = x.shape  # Save original shape
        x_flat = x.view(-1, self.embedding_dim)
        transformed = self.linear(x_flat)
        transformed = transformed.view(-1, self.num_chunks, self.embedding_dim)
        # Compute elementwise interactions
        interaction = transformed[:, 0] * transformed[:, 1]
        out_flat = interaction.clone()
        for idx in range(2, self.num_chunks):
            interaction = interaction * transformed[:, idx]
            out_flat = out_flat + interaction
        output = x_flat + out_flat
        output = output.view(*orig_shape)
        return output


class GatedMLP(nn.Module):
    def __init__(self, model_dim, expansion_factor=4, activation=None):
        super().__init__()
        self.activation = activation if activation is not None else nn.SELU()
        hidden_dim = model_dim * expansion_factor
        self.proj = init_linear(nn.Linear(model_dim, hidden_dim * 2))
        self.out = init_linear(nn.Linear(hidden_dim, model_dim))

    def forward(self, x):
        x_proj = self.proj(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        gated = x_a * self.activation(x_b)
        return self.out(gated)
