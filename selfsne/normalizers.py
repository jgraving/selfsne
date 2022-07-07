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

from selfsne.utils import off_diagonal, logmeanexp, log_interpolate, stop_gradient


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
        return self.logmeanexp(x)


class MomentumNormalizer(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(momentum, gradient=False)

    def forward(self, logits):
        return logits - self.log_ema(
            off_diagonal(logits) if logits.dim() > 1 else logits
        )


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


class LearnedNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_normalizer = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits - self.log_normalizer


class ConstantNormalizer(nn.Module):
    def __init__(self, log_normalizer=0.0):
        super().__init__()
        self.register_buffer("log_normalizer", torch.zeros(1) + log_normalizer)

    def forward(self, logits):
        return logits - self.log_normalizer


NORMALIZERS = {
    "batch": BatchNormalizer,
    "gradient_batch": GradientBatchNormalizer,
    "momentum": MomentumNormalizer,
    "gradient_momentum": GradientMomentumNormalizer,
    "learn": LearnedNormalizer,
    "constant": ConstantNormalizer,
}
