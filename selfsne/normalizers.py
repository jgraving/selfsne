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


class LogEMA(nn.Module):
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
                input, log_batch_average = ctx.saved_tensors
                log_n = np.log(input.numel())
                log_average = (
                    self.log_moving_average if ema_backward else log_batch_average
                )
                grad_output = grad_output * input.exp() / log_average.add(log_n).exp()
                return grad_output if gradient else None

        self.logmeanexp = LogMeanExp.apply

    def forward(self, x):
        if self.training:
            return self.logmeanexp(x)
        else:
            return self.log_moving_average


class MomentumNormalizer(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(momentum, gradient=False)

    def forward(self, y, logits):
        return self.log_ema(off_diagonal(logits) if logits.dim() > 1 else logits)


class GradientMomentumNormalizer(MomentumNormalizer):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(momentum, gradient=True)


class BatchNormalizer(MomentumNormalizer):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(momentum, gradient=False, ema_forward=False)


class GradientBatchNormalizer(MomentumNormalizer):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(
            momentum, gradient=True, ema_forward=False, ema_backward=False
        )


class ConditionalNormalizer(nn.Module):
    def forward(self, y, logits):
        return stop_gradient(logmeanexp(remove_diagonal(logits), dim=-1, keepdim=True))


class GradientConditionalNormalizer(nn.Module):
    def forward(self, y, logits):
        return logmeanexp(remove_diagonal(logits), dim=-1, keepdim=True)


class LearnedNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_normalizer = nn.Parameter(torch.zeros(1))

    def forward(self, y, logits):
        return self.log_normalizer


class ConstantNormalizer(nn.Module):
    def __init__(self, log_normalizer=0.0):
        super().__init__()
        self.register_buffer("log_normalizer", torch.zeros(1) + log_normalizer)

    def forward(self, y, logits):
        return self.log_normalizer


class LearnedConditionalNormalizer(nn.Module):
    def __init__(self, encoder, gradient=False):
        super().__init__()
        self.encoder = encoder
        self.gradient = gradient

    def forward(self, y, logits):
        return self.encoder(y if self.gradient else stop_gradient(y))


class LogInterpolatedNormalizer(nn.Module):
    def __init__(self, normalizer_a, normalizer_b, alpha=0.5):
        super().__init__()
        self.normalizer_a = normalizer_a
        self.normalizer_b = normalizer_b
        self.register_buffer("alpha_logit", torch.zeros(1).add(alpha).logit())

    def forward(self, y, logits):
        return log_interpolate(
            self.normalizer_a(y, logits), self.normalizer_b(y, logits), self.alpha_logit
        )


class InterpolatedNormalizer(nn.Module):
    def __init__(self, normalizer_a, normalizer_b, alpha=0.5):
        super().__init__()
        self.normalizer_a = normalizer_a
        self.normalizer_b = normalizer_b
        self.alpha = alpha

    def forward(self, y, logits):
        return interpolate(
            self.normalizer_a(y, logits), self.normalizer_b(y, logits), self.alpha
        )


class AdditiveNormalizer(nn.Module):
    def __init__(self, normalizers):
        super().__init__()
        self.normalizers = nn.ModuleList(normalizers)

    def forward(self, y, logits):
        log_normalizer = 0
        for normalizer in self.normalizers:
            log_normalizer = log_normalizer + normalizer(y, logits)
        return log_normalizer


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
