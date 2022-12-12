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


class LogMovingAverage(nn.Module):
    def __init__(
        self, momentum=0.99, gradient=True, ema_forward=True, ema_backward=True
    ):
        super().__init__()
        self.register_buffer("log_moving_average", torch.zeros(1))
        self.register_buffer("momentum_logit", torch.zeros(1).add(momentum).logit())

        class LogMeanExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
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

    def forward(self, x):
        if self.training:
            return self.logmeanexp(x)
        else:
            return self.log_moving_average


class LogMovingAverageBaseline(nn.Module):
    def __init__(
        self, momentum=0.99, gradient=True, ema_forward=True, ema_backward=True
    ):
        super().__init__()
        self.log_ema = LogMovingAverage(
            momentum=momentum,
            gradient=gradient,
            ema_forward=ema_forward,
            ema_backward=ema_backward,
        )

    def forward(self, y, logits):
        return self.log_ema(off_diagonal(logits) if logits.dim() > 1 else logits)


def MomentumBaseline(momentum=0.99):
    return LogMovingAverageBaseline(momentum=momentum, gradient=False)


def GradientMomentumBaseline(momentum=0.99):
    return LogMovingAverageBaseline(momentum=momentum, gradient=True)


def BatchBaseline(momentum=0.99):
    return LogMovingAverageBaseline(
        momentum=momentum, gradient=False, ema_forward=False
    )


def GradientBatchBaseline(momentum=0.99):
    return LogMovingAverageBaseline(
        momentum=momentum, gradient=True, ema_forward=False, ema_backward=False
    )


class GradientConditionalBaseline(nn.Module):
    def forward(self, y, logits):
        return logmeanexp(remove_diagonal(logits), dim=-1, keepdim=True)


class ConditionalBaseline(GradientConditionalBaseline):
    def forward(self, y, logits):
        return logmeanexp(remove_diagonal(stop_gradient(logits)), dim=-1, keepdim=True)


class LearnedBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_baseline = nn.Parameter(torch.zeros(1))

    def forward(self, y, logits):
        return self.log_baseline


class ConstantBaseline(nn.Module):
    def __init__(self, baseline=0):
        super().__init__()
        self.register_buffer("baseline", baseline)

    def forward(self, y, logits):
        return self.baseline


class LearnedConditionalBaseline(nn.Module):
    def __init__(self, encoder, gradient=False):
        super().__init__()
        self.encoder = encoder
        self.gradient = gradient

    def forward(self, y, logits):
        return self.encoder(y if self.gradient else stop_gradient(y))


class LogInterpolatedBaseline(nn.Module):
    def __init__(self, baseline_a, baseline_b, alpha=0.5):
        super().__init__()
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b
        self.register_buffer("alpha_logit", torch.zeros(1).add(alpha).logit())

    def forward(self, y, logits):
        return log_interpolate(
            self.baseline_a(y, logits), self.baseline_b(y, logits), self.alpha_logit
        )


class InterpolatedBaseline(nn.Module):
    def __init__(self, baseline_a, baseline_b, alpha=0.5):
        super().__init__()
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b
        self.alpha = alpha

    def forward(self, y, logits):
        return interpolate(
            self.baseline_a(y, logits), self.baseline_b(y, logits), self.alpha
        )


class AdditiveBaseline(nn.Module):
    def __init__(self, baselines, mean=False):
        super().__init__()
        self.baselines = nn.ModuleList(baselines)
        self.scale = len(baselines) if mean else 1
        self.log_scale = np.log(self.scale)

    def forward(self, y, logits):
        log_baseline = 0
        for baseline in self.baselines:
            log_baseline = log_baseline + baseline(y, logits)
        return log_baseline / self.scale


class LogAdditiveBaseline(AdditiveBaseline):
    def forward(self, y, logits):
        log_baseline = torch.zeros((1), device=y.device)
        for baseline in self.baselines:
            log_baseline = torch.logaddexp(log_baseline, baseline(y, logits))
        return log_baseline - self.log_scale


BASELINES = {
    "batch": BatchBaseline,
    "gradient_batch": GradientBatchBaseline,
    "conditional": ConditionalBaseline,
    "gradient_conditional": GradientConditionalBaseline,
    "momentum": MomentumBaseline,
    "gradient_momentum": GradientMomentumBaseline,
    "learn": LearnedBaseline,
    "constant": ConstantBaseline,
}
