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


class LogMeanExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, log_moving_average):
        ctx.save_for_backward(input, log_moving_average)
        return logmeanexp(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, log_moving_average = ctx.saved_tensors
        log_n = np.log(input.numel())
        moving_average = log_moving_average.add(log_n).exp()
        return grad_output * input.exp() / moving_average, None


class LogEMA(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        log_moving_average = torch.zeros(1)
        self.register_buffer("log_moving_average", log_moving_average)
        momentum = torch.zeros(1) + momentum
        self.register_buffer("momentum_logit", momentum.logit())

    def momentum_update(self, log_mean_x):
        self.log_moving_average = log_interpolate(
            self.log_moving_average, stop_gradient(log_mean_x), self.momentum_logit
        )
        return self.log_moving_average

    def forward(self, log_x):
        if self.training:
            return self.momentum_update(logmeanexp(log_x))
        else:
            return self.log_moving_average


class GradientLogEMA(LogEMA):
    def forward(self, log_x):
        if self.training:
            log_mean_x = LogMeanExp.apply(log_x, self.log_moving_average)
            self.momentum_update(log_mean_x)
            return log_mean_x
        else:
            return self.log_moving_average


class MomentumNormalizer(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(momentum)

    def forward(self, logits):
        return logits - self.log_ema(
            off_diagonal(logits) if logits.dim() > 1 else logits
        )


class GradientMomentumNormalizer(MomentumNormalizer):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = GradientLogEMA(momentum)


class BatchNormalizer(nn.Module):
    def forward(self, logits):
        return logits - logmeanexp(off_diagonal(logits) if logits.dim() > 1 else logits)


class LearnedNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_normalizer = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits - self.log_normalizer


class ConstantNormalizer(nn.Module):
    def __init__(self, log_normalizer=0.0):
        super().__init__()
        log_normalizer = torch.zeros(1) + log_normalizer
        self.register_buffer("log_normalizer", log_normalizer)

    def forward(self, logits):
        return logits - self.log_normalizer


NORMALIZERS = {
    "batch": BatchNormalizer,
    "momentum": MomentumNormalizer,
    "gradient_momentum": GradientMomentumNormalizer,
    "learn": LearnedNormalizer,
    "constant": ConstantNormalizer,
}
