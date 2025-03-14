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
import torch.nn as nn
from selfsne.nn import (
    GatedMLP,
    PreLayerNorm,
    PreRMSNorm,
    PreDyT,
    SelfAttention,
    LatentCrossAttention,
)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim,
        depth,
        num_heads,
        num_latent_tokens,
        dropout=0.0,
        norm_type="layernorm",
        expansion_factor=4,
        activation=None,
    ):
        """
        Args:
            model_dim: Dimensionality of input tokens.
            depth: Number of encoder layers.
            num_heads: Number of attention heads.
            num_latent_tokens: Number of output tokens (latent tokens).
            dropout: Dropout probability.
            norm_type: Normalization type ("layernorm", "rmsnorm", or "dyt").
            expansion_factor: Factor for expanding dimensions in GatedMLP.
            activation: Optional activation function for the GatedMLP.
        """
        super().__init__()

        # Select the prenormalization wrapper.
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            Norm = PreLayerNorm
        elif norm_type == "rmsnorm":
            Norm = PreRMSNorm
        elif norm_type == "dyt":
            Norm = PreDyT
        else:
            raise ValueError("norm_type must be 'layernorm', 'rmsnorm', or 'dyt'")

        # Create a stack of encoder layers.
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(
                            model_dim,
                            SelfAttention(model_dim, num_heads, dropout=dropout),
                        ),
                        Norm(
                            model_dim,
                            GatedMLP(
                                model_dim,
                                expansion_factor=expansion_factor,
                                activation=activation,
                            ),
                        ),
                    ]
                )
            )

        # Latent cross-attention block compresses the tokens.
        self.latent_cross_attention = LatentCrossAttention(
            model_dim, num_heads, num_latent_tokens, dropout
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [batch, tokens, model_dim].
            mask: Optional mask for attention computations.
        Returns:
            A tensor of shape [batch, num_latent_tokens, model_dim].
        """
        for self_attn, mlp in self.layers:
            x = self_attn(x)
            x = mlp(x)

        return self.latent_cross_attention(x, mask=mask)


# Example usage:
if __name__ == "__main__":
    # Example input: batch size 2, 16 tokens, model_dim 64.
    x = torch.randn(2, 16, 64)
    encoder = TransformerEncoder(
        model_dim=64,
        depth=4,
        num_heads=8,
        num_latent_tokens=8,
        dropout=0.1,
        norm_type="dyt",  # can be "layernorm", "rmsnorm", or "dyt"
        expansion_factor=4,
        activation=None,  # or specify a custom activation like nn.ReLU()
    )
    output = encoder(x)
    print("Output shape:", output.shape)  # Expected: [2, 8, 64]
