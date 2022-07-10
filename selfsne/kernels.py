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


def laplace(x, y, scale):
    return -(x - y).div_(scale).abs_().sum(-1)


def cauchy(x, y, scale):
    return -(x - y).div_(scale).pow_(2).sum(-1).log1p_()


def inverse(x, y, scale):
    return -(x - y).div_(scale).pow_(2).sum(-1).add(1e-5).log_()


def normal(x, y, scale):
    return -(x - y).div_(scale).pow_(2).div_(2).sum(-1)


def inner_product(x, y, scale):
    return (x * y).div_(scale).sum(-1)


def von_mises(x, y, scale):
    return inner_product(F.normalize(x, dim=-1), F.normalize(y, dim=-1), scale)


def wrapped_cauchy(x, y, scale):
    return -(np.cosh(scale) - von_mises(x, y, 1)).log()


def joint_product(x, y, scale):
    return (x.log_softmax(-1) + y.log_softmax(-1)).div_(scale).logsumexp(-1)


def bhattacharyya(x, y, scale):
    return joint_product(x, y, scale * 2)


KERNELS = {
    "normal": normal,
    "student_t": cauchy,
    "cauchy": cauchy,
    "inverse": inverse,
    "laplace": laplace,
    "von_mises": von_mises,
    "wrapped_cauchy": wrapped_cauchy,
    "inner_product": inner_product,
    "bhattacharyya": bhattacharyya,
    "joint_product": joint_product,
}
