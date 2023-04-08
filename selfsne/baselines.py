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

from selfsne.utils import (
    off_diagonal,
    remove_diagonal,
    logmeanexp,
    log_interpolate,
    stop_gradient,
)

from typing import Optional, List, Union


class LogMovingAverage(nn.Module):
    def __init__(
        self,
        momentum: float = 0.9,
        gradient: bool = True,
        ema_forward: bool = True,
        ema_backward: bool = True,
    ) -> None:
        """
        LogMovingAverage module that calculates the running log-mean-exp of input tensor.

        Args:
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
            gradient (bool, optional): Whether to enable gradient calculation. Defaults to True.
            ema_forward (bool, optional): Whether to use exponential moving average during forward pass. Defaults to True.
            ema_backward (bool, optional): Whether to use exponential moving average during backward pass. Defaults to True.
        """
        super().__init__()
        self.register_buffer("log_moving_average", torch.zeros(1))
        self.register_buffer("momentum_logit", torch.zeros(1).add(momentum).logit())

        class LogMeanExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                """
                Computes the log-mean-exp of the input tensor.

                Args:
                    input (torch.Tensor): Input tensor.

                Returns:
                    torch.Tensor: Log-mean-exp of the input tensor.
                """
                log_batch_average = logmeanexp(input)
                ctx.save_for_backward(input, log_batch_average)
                self.log_moving_average.data = stop_gradient(
                    log_interpolate(
                        self.log_moving_average, log_batch_average, self.momentum_logit
                    )
                )
                return self.log_moving_average if ema_forward else log_batch_average

            @staticmethod
            def backward(ctx, grad_output):
                """
                Computes the gradients of the log-mean-exp with respect to the input tensor.

                Args:
                    ctx: Saved tensors from forward pass.
                    grad_output (torch.Tensor): Gradient tensor.

                Returns:
                    torch.Tensor: Gradient of the log-mean-exp with respect to the input tensor.
                """
                if gradient:
                    input, log_batch_average = ctx.saved_tensors
                    log_n = np.log(input.numel())
                    log_average = (
                        self.log_moving_average if ema_backward else log_batch_average
                    )
                    grad_output = (
                        grad_output * input.exp() / log_average.add(log_n).exp()
                    )
                    return grad_output
                else:
                    return None

        self.logmeanexp = LogMeanExp.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-mean-exp of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Log-mean-exp of the input tensor.
        """
        if self.training:
            return self.logmeanexp(x)
        else:
            return self.log_moving_average


class MomentumBaseline(LogMovingAverage):
    def __init__(
        self,
        momentum: float = 0.9,
        gradient: bool = True,
        ema_forward: bool = True,
        ema_backward: bool = True,
    ) -> None:
        """
        Calculates the momentum baseline for a given set of logits.

        Args:
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
        """
        super().__init__(
            momentum=momentum,
            gradient=True,
            ema_forward=True,
            ema_backward=True,
        )

    def forward(
        self,
        logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the momentum baseline for a given set of logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Momentum baseline for the given set of logits.
        """
        return super().forward(off_diagonal(logits))


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
        return logmeanexp(off_diagonal(logits))


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
        return logmeanexp(remove_diagonal(logits), dim=-1, keepdim=True)


class LearnedBaseline(nn.Module):
    def __init__(self, activation: Optional[nn.Module] = nn.LogSigmoid()):
        """
        Initializes a learned baseline module.

        Args:
            activation (nn.Module, optional): Activation function to apply to the output. If None is passed, uses nn.Identity. Defaults to nn.LogSigmoid().
        """
        super().__init__()
        self.log_baseline = nn.Parameter(torch.zeros(1))
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns the learned baseline.

        Returns:
            torch.Tensor: Learned baseline.
        """
        return self.activation(self.log_baseline)


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


class LearnedConditionalBaseline(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_input: bool = False,
        activation: Optional[nn.Module] = nn.LogSigmoid(),
    ):
        """
        Initializes a learned conditional baseline module.

        Args:
            encoder (nn.Module): Encoder module to encode the input tensor y.
            embedding_input (bool, optional): Whether to pass embedded data z_y to the encoder instead of data y. Defaults to False.
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
        Calculates the learned conditional baseline for a given set of data y or embedded data z_y.

        Args:
            y (torch.Tensor, optional): Data input to the encoder. Ignored if embedding_input=True.
            z_y (torch.Tensor, optional): Embedded data input to the encoder. Used if embedding_input=True.

        Returns:
            torch.Tensor: Learned conditional baseline for the data y or z_y.
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
    "learn": LearnedBaseline,
    "constant": ConstantBaseline,
}
