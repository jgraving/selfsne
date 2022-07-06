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
    def __init__(self, momentum=0.99, gradient=True):
        super().__init__()
        self.register_buffer("log_moving_average", torch.zeros(1))
        momentum = torch.zeros(1) + momentum
        self.register_buffer("momentum_logit", momentum.logit())

        class LogMeanExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                self.log_moving_average.data = stop_gradient(
                    log_interpolate(
                        self.log_moving_average, logmeanexp(input), self.momentum_logit
                    )
                )
                return self.log_moving_average

            @staticmethod
            def backward(ctx, grad_output):
                input = ctx.saved_tensors[0]
                log_n = np.log(input.numel())
                moving_average = self.log_moving_average.add(log_n).exp()
                grad_output = grad_output * input.exp() / moving_average
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
        self.register_buffer("log_normalizer", torch.zeros(1) + log_normalizer)

    def forward(self, logits):
        return logits - self.log_normalizer


NORMALIZERS = {
    "batch": BatchNormalizer,
    "momentum": MomentumNormalizer,
    "gradient_momentum": GradientMomentumNormalizer,
    "learn": LearnedNormalizer,
    "constant": ConstantNormalizer,
}
