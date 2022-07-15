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


class LogMovingAverageNormalizer(nn.Module):
    def __init__(
        self, momentum=0.99, gradient=True, ema_forward=True, ema_backward=True
    ):
        super().__init__()
        self.pos_ema = LogMovingAverage(
            momentum=momentum,
            gradient=gradient,
            ema_forward=ema_forward,
            ema_backward=ema_backward,
        )
        self.neg_ema = LogMovingAverage(
            momentum=momentum,
            gradient=gradient,
            ema_forward=ema_forward,
            ema_backward=ema_backward,
        )

    def forward(self, y, pos_logits, neg_logits):
        return (self.pos_ema(pos_logits), self.neg_ema(neg_logits))


def MomentumNormalizer(momentum=0.99):
    return LogMovingAverageNormalizer(momentum=momentum, gradient=False)


def GradientMomentumNormalizer(momentum=0.99):
    return LogMovingAverageNormalizer(momentum=momentum, gradient=True)


def BatchNormalizer(momentum=0.99):
    return LogMovingAverageNormalizer(
        momentum=momentum, gradient=False, ema_forward=False
    )


def GradientBatchNormalizer(momentum=0.99):
    return LogMovingAverageNormalizer(
        momentum=momentum, gradient=True, ema_forward=False, ema_backward=False
    )


class GradientConditionalNormalizer(nn.Module):
    def forward(self, y, pos_logits, neg_logits):
        return (
            logmeanexp(pos_logits, dim=-1, keepdim=True),
            logmeanexp(neg_logits, dim=-1, keepdim=True),
        )


class ConditionalNormalizer(GradientConditionalNormalizer):
    def forward(self, y, pos_logits, neg_logits):
        return (
            logmeanexp(stop_gradient(pos_logits), dim=-1, keepdim=True),
            logmeanexp(stop_gradient(neg_logits), dim=-1, keepdim=True),
        )


class LearnedNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_normalizer = nn.Parameter(torch.zeros(1))

    def forward(self, y, pos_logits, neg_logits):
        return (self.log_normalizer, self.log_normalizer)


class ConstantNormalizer(nn.Module):
    def __init__(self, normalizer=1):
        super().__init__()
        self.register_buffer("log_normalizer", torch.zeros(1) + np.log(normalizer))

    def forward(self, y, pos_logits, neg_logits):
        return (self.log_normalizer, self.log_normalizer)


class LearnedConditionalNormalizer(nn.Module):
    def __init__(self, encoder, gradient=False):
        super().__init__()
        self.encoder = encoder
        self.gradient = gradient

    def forward(self, y, pos_logits, neg_logits):
        log_normalizer = self.encoder(y if self.gradient else stop_gradient(y))
        return (log_normalizer, log_normalizer)


class LogInterpolatedNormalizer(nn.Module):
    def __init__(self, normalizer_a, normalizer_b, alpha=0.5):
        super().__init__()
        self.normalizer_a = normalizer_a
        self.normalizer_b = normalizer_b
        self.alpha = alpha
        self.register_buffer("alpha_logit", torch.zeros(1).add(alpha).logit())

    def forward(self, y, pos_logits, neg_logits):
        pos_log_normalizer_a, neg_log_normalizer_a = self.normalizer_a(
            y, pos_logits, neg_logits
        )
        pos_log_normalizer_b, neg_log_normalizer_b = self.normalizer_b(
            y, pos_logits, neg_logits
        )
        return (
            log_interpolate(
                pos_log_normalizer_a, pos_log_normalizer_b, self.alpha_logit
            ),
            log_interpolate(
                neg_log_normalizer_a, neg_log_normalizer_b, self.alpha_logit
            ),
        )


class InterpolatedNormalizer(LogInterpolatedNormalizer):
    def forward(self, y, logits):
        pos_log_normalizer_a, neg_log_normalizer_a = self.normalizer_a(
            y, pos_logits, neg_logits
        )
        pos_log_normalizer_b, neg_log_normalizer_b = self.normalizer_b(
            y, pos_logits, neg_logits
        )
        return (
            interpolate(pos_log_normalizer_a, pos_log_normalizer_b, self.alpha),
            interpolate(neg_log_normalizer_a, neg_log_normalizer_b, self.alpha),
        )


class AdditiveNormalizer(nn.Module):
    def __init__(self, normalizers, mean=False):
        super().__init__()
        self.normalizers = nn.ModuleList(normalizers)
        self.scale = len(normalizers) if mean else 1
        self.log_scale = np.log(self.scale)

    def forward(self, y, pos_logits, neg_logits):
        pos_log_normalizer, neg_log_normalizer = 0, 0
        for normalizer in self.normalizers:
            add_pos_log_normalizer, add_neg_log_normalizer = normalizer(
                y, pos_logits, neg_logits
            )
            pos_log_normalizer = pos_log_normalizer + add_pos_log_normalizer
            neg_log_normalizer = neg_log_normalizer + add_neg_log_normalizer
        return (pos_log_normalizer / self.scale, neg_log_normalizer / self.scale)


class LogAdditiveNormalizer(AdditiveNormalizer):
    def forward(self, y, pos_logits, neg_logits):
        pos_log_normalizer = torch.zeros((1), device=y.device)
        neg_log_normalizer = torch.zeros((1), device=y.device)
        for normalizer in self.normalizers:
            add_pos_log_normalizer, add_neg_log_normalizer = normalizer(
                y, pos_logits, neg_logits
            )
            pos_log_normalizer = torch.logaddexp(
                pos_log_normalizer, add_pos_log_normalizer
            )
            neg_log_normalizer = torch.logaddexp(
                neg_log_normalizer, add_neg_log_normalizer
            )
        return (
            pos_log_normalizer - self.log_scale,
            neg_log_normalizer - self.log_scale,
        )


NORMALIZERS = {
    "batch": BatchNormalizer,
    "gradient_batch": GradientBatchNormalizer,
    "conditional": ConditionalNormalizer,
    "gradient_conditional": GradientConditionalNormalizer,
    "momentum": MomentumNormalizer,
    "gradient_momentum": GradientMomentumNormalizer,
    "learn": LearnedNormalizer,
    "constant": ConstantNormalizer,
}
