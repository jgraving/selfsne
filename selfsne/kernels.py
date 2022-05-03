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
from torch.distributions.utils import broadcast_all
import numpy as np


class Laplace(Module):
    def __init__(self, loc, scale=1.0):
        super().__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.abs().sum(-1)


class LaplaceT(Module):
    def __init__(self, loc, scale=1.0):
        super().__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.abs().sum(-1).log1p()


class StudentT(Module):
    def __init__(self, loc, scale=1.0):
        super().__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.pow(2).sum(-1).log1p()


class Normal(Module):
    def __init__(self, loc, scale=1.0):
        super().__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.pow(2).div(2).sum(-1)


class VonMises(Module):
    def __init__(self, loc):
        super().__init__()
        self.loc = loc / loc.norm(dim=-1, keepdim=True)

    def log_prob(self, value):
        value = value / value.norm(dim=-1, keepdim=True)
        return (self.loc * value).sum(-1)


class SphericalT(Module):
    def __init__(self, loc):
        super().__init__()
        self.loc = loc / loc.norm(dim=-1, keepdim=True)

    def log_prob(self, value):
        value = value / value.norm(dim=-1, keepdim=True)
        return -F.softplus(-(self.loc * value).sum(-1))


class Categorical(Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits.log_softmax(-1)

    def log_prob(self, value):
        return (self.logits * value.softmax(-1)).sum(-1)


class InnerProduct(Module):
    def __init__(self, loc):
        super().__init__()
        self.loc = loc

    def log_prob(self, value):
        return (self.loc * value).sum(-1)


class Bhattacharyya(Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits.log_softmax(-1)

    def log_prob(self, value):
        return 0.5 * (self.logits + value.log_softmax(-1)).logsumexp(-1)


KERNELS = {
    "normal": Normal,
    "studentt": StudentT,
    "categorical": Categorical,
    "laplace": Laplace,
    "vonmises": VonMises,
    "sphericalt": SphericalT,
    "laplacet": LaplaceT,
    "inner_product": InnerProduct,
    "bhattacharyya": Bhattacharyya,
}