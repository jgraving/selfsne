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

from selfsne.utils import (
    off_diagonal,
    logmeanexp,
    log_interpolate,
    stop_gradient,
)


class MomentumNormalizer(nn.Module):
    def __init__(self, momentum=0.9):
        super().__init__()
        log_normalizer = torch.zeros(1)
        self.register_buffer("log_normalizer", log_normalizer)
        momentum = torch.zeros(1) + momentum
        self.register_buffer("momentum_logit", momentum.logit())

    def momentum_update(self, log_normalizer):
        self.log_normalizer = log_interpolate(
            self.log_normalizer, stop_gradient(log_normalizer), self.momentum_logit
        )
        return self.log_normalizer

    def forward(self, logits):
        if self.training:
            log_normalizer = self.momentum_update(
                logmeanexp(off_diagonal(logits) if logits.dim() > 1 else logits)
            )
        else:
            log_normalizer = self.log_normalizer
        return logits - log_normalizer


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
    "momentum": MomentumNormalizer,
    "learn": LearnedNormalizer,
    "constant": ConstantNormalizer,
}
