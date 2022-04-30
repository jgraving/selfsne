# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
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

import torch.distributions as D
from torch.nn import Module


class Noise(Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            p = D.Bernoulli(probs=self.p).sample()
            if p == 1:
                return self._forward(x)
            else:
                return x
        else:
            return x


class Normal(Noise):
    def __init__(self, scale=1.0, p=1.0):
        super().__init__(p)
        self.scale = scale

    def _forward(self, x):
        return D.Normal(x, self.scale).rsample()


class LogNormalSoftmax(Normal):
    def _forward(self, x):
        return super()._forward(x.log().log_softmax(-1)).softmax(-1)


class GumbelSoftmax(Noise):
    def __init__(self, temperature=1.0, p=1.0):
        super().__init__(p)
        self.temperature = temperature

    def _forward(self, x):
        return D.RelaxedOneHotCategorical(self.temperature, logits=x.log()).rsample()


class ConditionalDropout(Noise):
    def __init__(self, p=1.0):
        super().__init__(p)

    def _forward(self, x):
        return x * D.Bernoulli(probs=x).sample()
