# -*- coding: utf-8 -*-
# Copyright 2020 Jacob M. Graving <jgraving@gmail.com>
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

import numpy as np

from selfsne.nn import init_selu
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

    def forward(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the momentum baseline for a given set of logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Momentum baseline for the given set of logits.
        """
        if self.training:
            return LogMeanExp.apply(
                logits,
                self.log_moving_average,
                self.momentum_logit,
            )
        else:
            return self.log_moving_average


class BatchBaseline(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the batch baseline for a given set of logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Batch baseline for the given set of logits.
        """
        return logmeanexp(logits)


class BatchConditionalBaseline(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the batch conditional baseline for a given set of logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Batch conditional baseline for the given set of logits.
        """
        return logmeanexp(logits, dim=-1, keepdim=True)


class ParametricBaseline(nn.Module):
    def __init__(
        self, param_dim: int = 256, activation: Optional[nn.Module] = nn.LogSigmoid()
    ):
        """
        Initializes a parametric baseline module.

        Args:
            activation (nn.Module, optional): Activation function to apply to the output. If None is passed, uses nn.Identity. Defaults to nn.LogSigmoid().
        """
        super().__init__()
        self.param = nn.Parameter(torch.randn(1, param_dim))
        self.projection = init_selu(nn.Linear(param_dim, 1))
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
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

    def forward(
        self,
        logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the constant baseline for a given set of logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Constant baseline for the given set of logits.
        """
        return self.baseline


class ConditionalBaseline(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_input: bool = True,
        activation: Optional[nn.Module] = nn.LogSigmoid(),
    ):
        """
        Initializes a parametric conditional baseline module.

        Args:
            encoder (nn.Module): Encoder module to encode the input tensor y.
            embedding_input (bool, optional): Whether to pass embedded data z_y to the encoder instead of data y. Defaults to True.
            activation (nn.Module, optional): Activation function to apply to the output. If None is passed, uses nn.Identity. Defaults to nn.LogSigmoid().
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_input = embedding_input
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(
        self,
        y: Optional[torch.Tensor] = None,
        z_y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the parametric conditional baseline for a given set of data y or embedded data z_y.

        Args:
            y (torch.Tensor, optional): Data input to the encoder. Ignored if embedding_input=True.
            z_y (torch.Tensor, optional): Embedded data input to the encoder. Used if embedding_input=True.

        Returns:
            torch.Tensor: Parametric conditional baseline for the data y or z_y.
        """
        if self.embedding_input:
            assert z_y is not None, "z_y must be provided when embedding_input=True"
            encoded = self.encoder(z_y)
        else:
            assert y is not None, "y must be provided when embedding_input=False"
            encoded = self.encoder(y)
        return self.activation(encoded)


class LogInterpolatedBaseline(nn.Module):
    def __init__(
        self, baseline_a: nn.Module, baseline_b: nn.Module, alpha: float = 0.5
    ):
        """
        Initializes a logarithmically interpolated baseline module.

        Args:
            baseline_a (nn.Module): First baseline module.
            baseline_b (nn.Module): Second baseline module.
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__()
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b
        self.register_buffer("alpha_logit", torch.tensor(alpha).logit())

    def forward(
        self,
        logits: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the logarithmically interpolated baseline.

        Args:
            logits (torch.Tensor): Logits tensor.
            y (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_y (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Logarithmically interpolated baseline for the given set of logits and optional data y.
        """
        return log_interpolate(
            self.baseline_a(logits=logits, y=y, z_y=z_y),
            self.baseline_b(logits=logits, y=y, z_y=z_y),
            self.alpha_logit,
        )


class InterpolatedBaseline(nn.Module):
    def __init__(
        self, baseline_a: nn.Module, baseline_b: nn.Module, alpha: float = 0.5
    ):
        """
        Initializes an interpolated baseline module.

        Args:
            baseline_a (nn.Module): First baseline module.
            baseline_b (nn.Module): Second baseline module.
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__()
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the interpolated baseline.

        Args:
            logits (torch.Tensor): Logits tensor.
            y (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_y (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Interpolated baseline for the given set of logits and optional data y.
        """
        return interpolate(
            self.baseline_a(logits=logits, y=y, z_y=z_y),
            self.baseline_b(logits=logits, y=y, z_y=z_y),
            self.alpha,
        )


class AdditiveBaseline(nn.Module):
    def __init__(self, baselines: List[nn.Module], mean: bool = False):
        """
        Initializes an additive baseline module.

        Args:
            baselines (List[nn.Module]): List of baseline modules to be added together.
            mean (bool, optional): Whether to take the mean of the baselines instead of summing them. Defaults to False.
        """
        super().__init__()
        self.baselines = nn.ModuleList(baselines)
        self.scale = len(baselines) if mean else 1
        self.log_scale = np.log(self.scale)

    def forward(
        self,
        logits: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the additive baseline.

        Args:
            logits (torch.Tensor): Logits tensor.
            y (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_y (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Additive baseline for the given set of logits and optional data y.
        """
        log_baseline = 0
        for baseline in self.baselines:
            log_baseline = log_baseline + baseline(logits=logits, y=y, z_y=z_y)
        return log_baseline / self.scale


class LogAdditiveBaseline(AdditiveBaseline):
    def __init__(self, baselines: List[nn.Module], mean: bool = False):
        """
        Initializes a logarithmic additive baseline module.

        Args:
            baselines (List[nn.Module]): List of baseline modules to be added together.
            mean (bool, optional): Whether to take the mean of the baselines instead of summing them. Defaults to False.
        """
        super().__init__(baselines, mean)

    def forward(
        self,
        logits: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the logarithmic additive baseline for a given set of logits and optional data y.

        Args:
            logits (torch.Tensor): Logits tensor.
            y (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_y (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Logarithmic additive baseline for the given set of logits and optional data y.
        """
        log_baseline = torch.zeros((1), device=logits.device)
        for baseline in self.baselines:
            log_baseline = torch.logaddexp(
                log_baseline, baseline(logits=logits, y=y, z_y=z_y)
            )
        return log_baseline - self.log_scale


BASELINES = {
    "batch": BatchBaseline,
    "batch_conditional": BatchConditionalBaseline,
    "momentum": MomentumBaseline,
    "parametric": ParametricBaseline,
    "constant": ConstantBaseline,
}
