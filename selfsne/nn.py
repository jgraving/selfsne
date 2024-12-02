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
        return self.residual(x) + self.module(x)


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


class Scale(nn.Module):
    def __init__(self, param_dim: int = 1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1, param_dim))
        self.projection = init_selu(nn.Linear(param_dim, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.projection(self.param))
        return input * scale


class Bias(nn.Module):
    def __init__(self, param_dim: int = 1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1, param_dim))
        self.projection = init_selu(nn.Linear(param_dim, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.projection(self.param)
        return input + bias


class ScaleBias(nn.Module):
    def __init__(self, param_dim: int = 1024):
        super().__init__()
        self.scale_param = nn.Parameter(torch.randn(1, param_dim))
        self.bias_param = nn.Parameter(torch.randn(1, param_dim))
        self.scale_projection = init_selu(nn.Linear(param_dim, 1))
        self.bias_projection = init_selu(nn.Linear(param_dim, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.scale_projection(self.scale_param))
        bias = self.bias_projection(self.bias_param)
        return input * scale + bias


class PairSampler(nn.Module):
    def __init__(self, x_sampler=nn.Identity(), y_sampler=nn.Identity()):
        super().__init__()
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler

    def forward(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = None

        sampled_x = self.x_sampler(x)
        if y is None:
            sampled_y = self.y_sampler(x)
        else:
            sampled_y = self.y_sampler(y)
        return sampled_x, sampled_y


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


class EmbeddingEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, encoder: nn.Module):
        """
        Initialize the EmbeddingEncoder module.

        Args:
            num_embeddings (int): The size of the embedding dictionary.
            embedding_dim (int): The dimension of each embedding vector.
            encoder (nn.Module): The encoder module to encode the embedding parameters.
        """
        super(EmbeddingEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder = encoder

    def forward(self, dummy, **kwargs) -> torch.Tensor:
        """
        Forward pass through the embedding layer and then the encoder.

        Returns:
            torch.Tensor: The output after passing the embedding parameters through the encoder.
        """
        return self.encoder(self.embedding.weight)


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
        pos_embed = height_embed + width_embed
        # Repeat the pos_embed for the entire batch
        pos_embed = pos_embed.repeat(x.shape[0], 1, 1, 1)
        # Add the position embedding 2D to the input tensor
        return torch.cat([x, pos_embed], dim=1)


class PositionEmbedding(nn.Module):
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


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Dynamically computes and applies Sinusoidal Positional Embeddings
    to the input tokens. Assumes input tensor shape (batch_size, seq_len, model_dim).
    """

    def __init__(self):
        super(SinusoidalPositionalEmbedding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds sinusoidal positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Tensor with sinusoidal positional embeddings added, same shape as input.
        """
        batch_size, seq_len, model_dim = x.shape

        # Compute positions and div_term dynamically
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float, device=x.device)
            * -(math.log(10000.0) / model_dim)
        )  # (model_dim / 2)

        # Compute sinusoidal positional embeddings
        sin_emb = torch.sin(position * div_term)  # (seq_len, model_dim / 2)
        cos_emb = torch.cos(position * div_term)  # (seq_len, model_dim / 2)

        # Concatenate sine and cosine embeddings along the model dimension
        positional_embedding = torch.zeros(
            (seq_len, model_dim), device=x.device
        )  # (seq_len, model_dim)
        positional_embedding[:, 0::2] = sin_emb  # Assign sin to even indices
        positional_embedding[:, 1::2] = cos_emb  # Assign cos to odd indices

        # Add positional embeddings to input tensor
        positional_embedding = positional_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, seq_len, model_dim)
        return x + positional_embedding  # (batch_size, seq_len, model_dim)


class RotaryPositionalEmbedding(nn.Module):
    """
    Dynamically computes and applies Rotary Positional Embeddings (RoPE)
    to the input tokens. Assumes input tensor shape (batch_size, seq_len, model_dim).
    """

    def __init__(self):
        super(RotaryPositionalEmbedding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        batch_size, seq_len, model_dim = x.shape

        # Compute positions and div_term dynamically
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float, device=x.device)
            * -(math.log(10000.0) / model_dim)
        )  # (model_dim / 2)

        # Compute the sinusoidal positional embeddings
        sin_emb = torch.sin(position * div_term).unsqueeze(
            0
        )  # (1, seq_len, model_dim / 2)
        cos_emb = torch.cos(position * div_term).unsqueeze(
            0
        )  # (1, seq_len, model_dim / 2)

        # Apply RoPE
        x1 = (
            x[..., 0::2] * cos_emb - x[..., 1::2] * sin_emb
        )  # (batch_size, seq_len, model_dim / 2)
        x2 = (
            x[..., 0::2] * sin_emb + x[..., 1::2] * cos_emb
        )  # (batch_size, seq_len, model_dim / 2)

        return torch.cat([x1, x2], dim=-1)  # (batch_size, seq_len, model_dim)


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

        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.out = nn.Linear(model_dim, model_dim)
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
    def __init__(self, model_dim, num_heads, num_latent_tokens, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(1, num_latent_tokens, model_dim))

        # Linear layers for Q, K, V
        self.q = nn.Linear(model_dim, model_dim)
        self.kv = nn.Linear(model_dim, model_dim * 2)
        self.out = nn.Linear(model_dim, model_dim)

        # Attention module
        self.attention = Attention(model_dim, num_heads, dropout)

    def forward(self, x):
        batch_size, _, _ = x.shape

        # Generate Q from latent tokens
        q = self.q(self.latents)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)
        q = q.expand(batch_size, -1, -1, -1)

        # Generate K, V from input
        kv = self.kv(x)
        kv = rearrange(
            kv, "b s (h d c) -> c b h s d", h=self.num_heads, d=self.head_dim, c=2
        )
        k, v = kv[0], kv[1]

        # Apply attention
        out = self.attention(q, k, v)

        # Rearrange and apply output linear layer
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out(out) + self.latents
