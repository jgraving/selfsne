# -*- coding: utf-8 -*-
# Copyright 2020 Jacob M. Graving <jgraving@gmail.com>
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
from torch.nn import init

import numpy as np

from selfsne.nn import init_selu
from selfsne.kernels import pairwise_inner_product
from selfsne.utils import (
    off_diagonal,
    remove_diagonal,
    logmeanexp,
    log_interpolate,
    stop_gradient,
)

from typing import Optional, List, Union


class LogMeanExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, log_moving_average, momentum_logit):
        log_batch_average = logmeanexp(input)
        updated_log_moving_average = stop_gradient(
            log_interpolate(log_moving_average, log_batch_average, momentum_logit)
        )
        log_moving_average.data = updated_log_moving_average
        ctx.save_for_backward(input, updated_log_moving_average)
        return updated_log_moving_average

    @staticmethod
    def backward(ctx, grad_output):
        input, log_moving_average = ctx.saved_tensors
        log_n = np.log(input.numel())
        # ratio = input.exp() / log_moving_average.add(log_n).exp()
        # Compute the ratio in the log domain
        ratio = torch.exp(input - (log_moving_average + log_n))
        return grad_output * ratio, None, None


class MomentumBaseline(nn.Module):
    def __init__(
        self,
        momentum: float = 0.9,
    ) -> None:
        """
        Calculates the momentum baseline for a given set of logits.

        Args:
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
        """
        super().__init__()
        self.register_buffer("log_moving_average", torch.zeros(1))
        self.register_buffer("momentum_logit", torch.zeros(1).add(momentum).logit())

    def forward(self, neg_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the momentum baseline.

        Args:
            neg_logits (torch.Tensor): Negative logits tensor.

        Returns:
            torch.Tensor: Momentum baseline.
        """
        if self.training:
            return LogMeanExp.apply(
                neg_logits, self.log_moving_average, self.momentum_logit
            )
        else:
            return self.log_moving_average


class ReverseMomentumBaseline(MomentumBaseline):
    def forward(self, pos_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the reverse momentum baseline.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.

        Returns:
            torch.Tensor: Reverse momentum baseline.
        """
        if self.training:
            return -LogMeanExp.apply(
                -pos_logits, self.log_moving_average, self.momentum_logit
            )
        else:
            return -self.log_moving_average


class BatchBaseline(nn.Module):
    def forward(self, neg_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the batch baseline.

        Args:
            neg_logits (torch.Tensor): Negative logits tensor.

        Returns:
            torch.Tensor: Batch baseline.
        """
        return logmeanexp(neg_logits)


class ReverseBatchBaseline(nn.Module):
    def forward(self, pos_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the reverse batch baseline.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.

        Returns:
            torch.Tensor: Reverse batch baseline.
        """
        return -logmeanexp(-pos_logits)


class BatchConditionalBaseline(nn.Module):
    def forward(self, neg_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the batch conditional baseline.

        Args:
            neg_logits (torch.Tensor): Negative logits tensor.

        Returns:
            torch.Tensor: Batch conditional baseline.
        """
        return logmeanexp(neg_logits, dim=-1, keepdim=True)


class ReverseBatchConditionalBaseline(nn.Module):
    def forward(self, pos_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the reverse batch conditional baseline.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.

        Returns:
            torch.Tensor: Reverse batch conditional baseline.
        """
        return -logmeanexp(-pos_logits, dim=-1, keepdim=True)


class ParametricBaseline(nn.Module):
    def __init__(
        self,
        param_dim: int = 1024,
        activation: Optional[nn.Module] = nn.Identity(),
        bias_init: float = 0.0,
    ):
        """
        Initializes a parametric baseline module.

        Args:
            param_dim (int, optional): Dimensionality of the parameter. Defaults to 1024.
            activation (nn.Module, optional): Activation function to apply. Defaults to nn.Identity().
            bias_init (float, optional): Bias initialization value. Defaults to 0.0.
        """
        super().__init__()
        self.param = nn.Parameter(torch.randn(1, param_dim))
        self.projection = init_selu(nn.Linear(param_dim, 1))
        init.constant_(self.projection.bias, bias_init)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Returns the parametric baseline.

        Returns:
            torch.Tensor: Parametric baseline.
        """
        return self.activation(self.projection(self.param))


class ConstantBaseline(nn.Module):
    def __init__(self, baseline: float = 0.0):
        """
        Initializes a constant baseline module.

        Args:
            baseline (float, optional): Constant baseline value. Defaults to 0.0.
        """
        super().__init__()
        self.register_buffer("baseline", torch.tensor(baseline))

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Returns the constant baseline.

        Returns:
            torch.Tensor: Constant baseline.
        """
        return self.baseline


class ParametricConditionalBaseline(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        param_dim: int = 1024,
        activation: Optional[nn.Module] = nn.Identity(),
        bias_init: float = 0.0,
    ):
        """
        Initializes a parametric conditional baseline module.

        Args:
            encoder (nn.Module): Encoder for the embedded context.
            param_dim (int, optional): Dimensionality of the parameter. Defaults to 1024.
            activation (nn.Module, optional): Activation function. Defaults to nn.Identity().
            bias_init (float, optional): Bias initialization value. Defaults to 0.0.
        """
        super().__init__()
        self.encoder = encoder
        self.parametric_baseline = ParametricBaseline(param_dim, bias_init=bias_init)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(
        self, context_embedding: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Calculates the parametric conditional baseline.

        Args:
            context_embedding (torch.Tensor, optional): Embedded context.

        Returns:
            torch.Tensor: Parametric conditional baseline.
        """
        encoded = self.encoder(context_embedding)
        scalar_baseline = self.parametric_baseline()
        return self.activation(scalar_baseline + encoded)


ConditionalBaseline = ParametricConditionalBaseline


class AdditiveBaseline(nn.Module):
    def __init__(self, baselines: List[nn.Module], mean: bool = False):
        """
        Initializes an additive baseline module.

        Args:
            baselines (List[nn.Module]): List of baseline modules.
            mean (bool, optional): Whether to average the baselines. Defaults to False.
        """
        super().__init__()
        self.baselines = nn.ModuleList(baselines)
        self.scale = len(baselines) if mean else 1
        self.log_scale = np.log(self.scale)

    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the additive baseline.

        Args:
            pos_logits (torch.Tensor, optional): Positive logits.
            neg_logits (torch.Tensor, optional): Negative logits.
            context (torch.Tensor, optional): Context input.
            context_embedding (torch.Tensor, optional): Embedded context.

        Returns:
            torch.Tensor: Additive baseline.
        """
        log_baseline = 0
        for baseline in self.baselines:
            log_baseline += baseline(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                context=context,
                context_embedding=context_embedding,
            )
        return log_baseline / self.scale


class LogAdditiveBaseline(AdditiveBaseline):
    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the logarithmic additive baseline.

        Args:
            pos_logits (torch.Tensor, optional): Positive logits.
            neg_logits (torch.Tensor, optional): Negative logits.
            context (torch.Tensor, optional): Context input.
            context_embedding (torch.Tensor, optional): Embedded context.

        Returns:
            torch.Tensor: Logarithmic additive baseline.
        """
        log_baseline = self.baselines[0](
            pos_logits=pos_logits, context=context, context_embedding=context_embedding
        )
        for baseline in self.baselines[1:]:
            additional_value = baseline(
                pos_logits=pos_logits,
                context=context,
                context_embedding=context_embedding,
            )
            log_baseline = torch.logaddexp(log_baseline, additional_value)
        return log_baseline - self.log_scale


class EmbeddingBaseline(nn.Module):
    """
    Computes a log baseline from a context embedding and a baseline embedding
    using a provided rowwise similarity kernel and kernel scale. This baseline
    is used to adjust logits in a contrastive loss by serving as a similarity threshold.

    The kernel and kernel_scale are passed as arguments during the forward pass.
    Additional keyword arguments are accepted for compatibility.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        context_embedding: torch.Tensor,
        baseline_embedding: torch.Tensor,
        kernel: callable,
        kernel_scale: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the log baseline as the similarity between the context embedding and
        the baseline embedding using the provided kernel and kernel scale.

        Args:
            context_embedding (torch.Tensor): The context embedding.
            baseline_embedding (torch.Tensor): The baseline embedding.
            kernel (callable): A rowwise similarity kernel function.
            kernel_scale (float): The scale factor for the kernel.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed log baseline.
        """
        log_baseline = kernel(context_embedding, baseline_embedding, kernel_scale)
        return log_baseline.unsqueeze(1)


BASELINES = {
    "batch": BatchBaseline,
    "batch_conditional": BatchConditionalBaseline,
    "momentum": MomentumBaseline,
    "reverse_batch": ReverseBatchBaseline,
    "reverse_batch_conditional": ReverseBatchConditionalBaseline,
    "reverse_momentum": ReverseMomentumBaseline,
    "parametric": ParametricBaseline,
    "constant": ConstantBaseline,
    "embedding": EmbeddingBaseline,
}
