# nn/sampling.py
# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from torch import nn
from selfsne.utils import random_sample_columns

__all__ = ["TokenSampler"]


class TokenSampler(nn.Module):
    """
    Samples a subset of tokens along the token dimension.

    When in training mode, this module randomly samples a subset of the input tokens
    based on a fixed number (`num_tokens`) or a fraction (`p`). In evaluation mode,
    the input is returned unchanged.
    """

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
        """
        Forward pass for token sampling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, tokens, features).

        Returns:
            torch.Tensor: Tensor with a subset of tokens sampled.
        """
        if self.training:
            batch_size, tokens, features = x.size()
            if self.num_tokens is not None:
                tokens_to_keep = min(tokens, self.num_tokens)
            elif self.p is not None:
                tokens_to_keep = int(tokens * self.p)
            else:
                return x
            output_tensor = random_sample_columns(x, tokens_to_keep)
            return output_tensor
        else:
            return x
