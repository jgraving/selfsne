# nn/attention.py
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
from einops import rearrange

from selfsne.nn.init import init_linear

__all__ = [
    "Attention",
    "SelfAttention",
    "CrossAttention",
    "LatentCrossAttention",
]


class Attention(nn.Module):
    """
    Implements scaled dot-product attention with optional causal masking.
    """

    def __init__(self, model_dim, num_heads, dropout=0.0, causal_mask=False):
        super().__init__()
        assert (
            model_dim % num_heads == 0
        ), f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.causal_mask = causal_mask

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, num_heads, seq_len, head_dim)
        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if self.causal_mask:
            seq_len = similarity.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=similarity.device), diagonal=1
            ).bool()
            similarity = similarity.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            similarity = similarity.masked_fill(
                mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attention = similarity.softmax(dim=-1)
        attention = self.dropout(attention)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        return out


class SelfAttention(nn.Module):
    """
    Self-attention mechanism that computes queries, keys, and values from the same input.
    """

    def __init__(self, model_dim, num_heads, dropout=0.0, causal_mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.qkv = init_linear(nn.Linear(model_dim, model_dim * 3))
        self.out = init_linear(nn.Linear(model_dim, model_dim))
        self.attention = Attention(model_dim, num_heads, dropout, causal_mask)

    def forward(self, x):
        # x: (batch_size, seq_len, model_dim)
        qkv = self.qkv(x)  # Shape: (batch_size, seq_len, 3 * model_dim)
        qkv = rearrange(
            qkv, "b s (c h d) -> c b h s d", c=3, h=self.num_heads, d=self.head_dim
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = self.attention(q, k, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for attending from one sequence (query) to another (context).
    """

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q = init_linear(nn.Linear(model_dim, model_dim))
        self.kv = init_linear(nn.Linear(model_dim, model_dim * 2))
        self.out = init_linear(nn.Linear(model_dim, model_dim))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.attention = Attention(model_dim, num_heads, dropout)

    def forward(self, query, context, mask=None):
        # query: (batch_size, query_len, model_dim)
        # context: (batch_size, context_len, model_dim)
        q = self.q(query)  # (batch_size, query_len, model_dim)
        kv = self.kv(context)  # (batch_size, context_len, 2 * model_dim)
        kv = rearrange(
            kv, "b s (c h d) -> c b h s d", c=2, h=self.num_heads, d=self.head_dim
        )
        k, v = kv[0], kv[1]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        out = self.attention(q, k, v, mask=mask)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class LatentCrossAttention(nn.Module):
    """
    Uses learnable latent tokens as queries to perform cross-attention over context tokens.
    """

    def __init__(self, model_dim, num_heads, num_latent_tokens, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.latents = nn.Parameter(
            torch.randn(1, num_latent_tokens, model_dim) / np.sqrt(model_dim)
        )
        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, context_len, model_dim)
        batch_size = x.size(0)
        latents = self.latents.expand(batch_size, -1, -1)
        return self.cross_attention(latents, x, mask=mask) + latents
