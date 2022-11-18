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
import torch
import numpy as np


def laplace(x, y, scale):
    return torch.pdist(x, y, p=1).div(scale).neg()


def pairwise_laplace(x, y, scale):
    return torch.cdist(x, y.squeeze(), p=1).div(scale).neg()


def cauchy(x, y, scale):
    return torch.pdist(x, y).div(scale).pow(2).log1p().neg()


def pairwise_cauchy(x, y, scale):
    return torch.cdist(x, y.squeeze()).div(scale).pow(2).log1p().neg()


def inverse(x, y, scale):
    return torch.pdist(x, y).div(scale).pow(2).add(1e-5).log()


def pairwise_inverse(x, y, scale):
    return torch.cdist(x, y.squeeze()).div(scale).pow(2).add(1e-5).log()


def normal(x, y, scale):
    return torch.pdist(x - y).div(scale).pow(2).div(2).neg()


def pairwise_normal(x, y, scale):
    return torch.cdist(x, y.squeeze()).div(scale).pow(2).neg()


def inner_product(x, y, scale):
    return (x * y).sum(-1).div(scale)


def pairwise_inner_product(x, y, scale):
    return (x @ y.squeeze().T).div(scale)


def von_mises(x, y, scale):
    return inner_product(F.normalize(x, dim=-1), F.normalize(y, dim=-1), scale)


def pairwise_von_mises(x, y, scale):
    return pairwise_inner_product(F.normalize(x, dim=-1), F.normalize(y, dim=-1), scale)


def wrapped_cauchy(x, y, scale):
    return (np.cosh(scale) - von_mises(x, y, 1)).log().neg()


def pairwise_wrapped_cauchy(x, y, scale):
    return (np.cosh(scale) - pairwise_von_mises(x, y, 1)).log().neg()


def joint_product(x, y, scale):
    return (x.log_softmax(-1) + y.log_softmax(-1)).div_(scale).logsumexp(-1)


def bhattacharyya(x, y, scale):
    return joint_product(x, y, scale * 2)


KERNELS = {
    "normal": normal,
    "pairwise_normal": pairwise_normal,
    "student_t": cauchy,
    "pairwise_student_t": pairwise_cauchy,
    "cauchy": cauchy,
    "pairwise_cauchy": pairwise_cauchy,
    "inverse": inverse,
    "pairwise_inverse": pairwise_inverse,
    "laplace": laplace,
    "pairwise_laplace": pairwise_laplace,
    "von_mises": von_mises,
    "pairwise_von_mises": pairwise_von_mises,
    "wrapped_cauchy": wrapped_cauchy,
    "pairwise_wrapped_cauchy": pairwise_wrapped_cauchy,
    "inner_product": inner_product,
    "pairwise_inner_product": pairwise_inner_product,
    "bhattacharyya": bhattacharyya,
    "joint_product": joint_product,
}
