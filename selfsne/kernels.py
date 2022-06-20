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

from torch.nn import Module
import torch.nn.functional as F
import numpy as np


class LocScale(Module):
    def __init__(self, loc, scale=1.0):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, value):
        return (value - self.loc).div_(self.scale)


class Laplace(LocScale):
    def log_prob(self, value):
        return -self(value).abs_().sum(-1)


class LaplaceT(LocScale):
    def log_prob(self, value):
        return -self(value).abs_().sum(-1).log1p_()


class StudentT(LocScale):
    def log_prob(self, value):
        return -self(value).pow_(2).sum(-1).log1p_()


class Inverse(LocScale):
    def log_prob(self, value):
        return -self(value).pow_(2).sum(-1).add(1e-5).log_()


class Normal(LocScale):
    def log_prob(self, value):
        return -self(value).pow_(2).div_(2).sum(-1)


class InnerProduct(LocScale):
    def forward(self, value):
        return (self.loc * value).div_(self.scale).sum(-1)

    def log_prob(self, value):
        return self(value)


class VonMises(InnerProduct):
    def __init__(self, loc, scale=1.0):
        super().__init__(F.normalize(loc, dim=-1), scale)

    def log_prob(self, value):
        return self(F.normalize(value, dim=-1))


class WrappedCauchy(VonMises):
    def log_prob(self, value):
        cos = (self.loc * F.normalize(value, dim=-1)).sum(-1)
        return -(np.cosh(self.scale) - cos).log()


class Categorical(Module):
    def __init__(self, logits, temperature):
        super().__init__()
        self.logits = logits.log_softmax(-1)
        self.temperature = temperature

    def log_prob(self, value):
        return (self.logits * value.softmax(-1)).div_(self.temperature).sum(-1)


class JointProduct(Categorical):
    def forward(self, value):
        return (self.logits + value.log_softmax(-1)).div_(self.temperature)

    def log_prob(self, value):
        return self(value).logsumexp(-1)


class Bhattacharyya(JointProduct):
    def log_prob(self, value):
        return self(value).mul_(0.5).logsumexp(-1)


KERNELS = {
    "normal": Normal,
    "studentt": StudentT,
    "cauchy": StudentT,
    "inverse": Inverse,
    "categorical": Categorical,
    "laplace": Laplace,
    "vonmises": VonMises,
    "wrapped_cauchy": WrappedCauchy,
    "laplacet": LaplaceT,
    "inner_product": InnerProduct,
    "bhattacharyya": Bhattacharyya,
    "joint_product": JointProduct,
}
