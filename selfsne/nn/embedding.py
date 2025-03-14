# nn/embedding.py
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

import math
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from selfsne.nn.init import init_linear

__all__ = [
    "PositionEmbedding2d",
    "PositionEmbedding1d",
    "SinusoidalPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "PatchEmbedding1d",
    "PatchEmbedding2d",
]


class PositionEmbedding2d(nn.Module):
    """
    Adds a learned 2D positional embedding to the input by concatenating
    separate height and width embeddings.
    """

    def __init__(self, embedding_dim, height, width):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.height = height
        self.width = width
        self.height_embedding = nn.Parameter(torch.randn(embedding_dim, height))
        self.width_embedding = nn.Parameter(torch.randn(embedding_dim, width))

    def forward(self, x):
        # Create a positional embedding tensor of shape (1, embedding_dim, height, width)
        height_embed = self.height_embedding.view(1, self.embedding_dim, self.height, 1)
        width_embed = self.width_embedding.view(1, self.embedding_dim, 1, self.width)
        pos_embed = height_embed + width_embed
        pos_embed = pos_embed.repeat(x.shape[0], 1, 1, 1)
        # Concatenate along the channel dimension
        return torch.cat([x, pos_embed], dim=1)


class PositionEmbedding1d(nn.Module):
    """
    Adds a learned 1D positional embedding to the input.
    """

    def __init__(self, embedding_dim, num_positions):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_positions, embedding_dim))

    def forward(self, x):
        # x is assumed to be of shape (batch_size, seq_len, embedding_dim)
        return x + self.embedding[: x.shape[1]]


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Computes sinusoidal positional embeddings and adds them to the input.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.register_buffer(
            "div_term",
            torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * -(math.log(10000.0) / model_dim)
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        assert (
            model_dim == self.model_dim
        ), "Input model_dim must match initialized model_dim."
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        sin_emb = torch.sin(position * self.div_term)
        cos_emb = torch.cos(position * self.div_term)
        # Rearranging the sine and cosine embeddings into a single positional embedding
        positional_embedding = rearrange(
            [sin_emb, cos_emb], "d seq half_dim -> seq (d half_dim)"
        )
        positional_embedding = rearrange(
            positional_embedding, "seq dim -> 1 seq dim"
        ).expand(batch_size, -1, -1)
        return x + positional_embedding


class RotaryPositionalEmbedding(nn.Module):
    """
    Applies Rotary Positional Embedding (RoPE) to the input.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.register_buffer(
            "div_term",
            torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * -(math.log(10000.0) / model_dim)
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        assert (
            model_dim == self.model_dim
        ), "Input model_dim must match initialized model_dim."
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        sin_emb = rearrange(
            torch.sin(position * self.div_term), "seq half_dim -> 1 seq half_dim"
        )
        cos_emb = rearrange(
            torch.cos(position * self.div_term), "seq half_dim -> 1 seq half_dim"
        )
        # Split the input into even and odd indices
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x1 = x_even * cos_emb - x_odd * sin_emb
        x2 = x_even * sin_emb + x_odd * cos_emb
        # Concatenate the transformed parts back together
        return rearrange([x1, x2], "d b seq half_dim -> b seq (d half_dim)")


def PatchEmbedding1d(seq_length, patch_size, embedding_dim, in_channels):
    """
    Splits a 1D sequence into patches and projects them into an embedding space.
    """
    assert (
        seq_length % patch_size == 0
    ), "Sequence length must be divisible by patch size."
    return nn.Sequential(
        init_linear(
            nn.Conv1d(
                in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        ),
        Rearrange("b c s -> b s c"),
    )


def PatchEmbedding2d(image_size, patch_size, embedding_dim, in_channels=3):
    """
    Splits a 2D image into patches and projects them into an embedding space.
    """
    assert image_size % patch_size == 0, "Image size must be divisible by patch size."
    return nn.Sequential(
        init_linear(
            nn.Conv2d(
                in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
            )
        ),
        Rearrange("b c h w -> b (h w) c"),
    )
