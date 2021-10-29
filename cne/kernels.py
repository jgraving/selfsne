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

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.nn import Module
from torch.distributions.utils import broadcast_all

import numpy as np


def inv_softplus(x):
    return x.expm1().log()


class LaplaceKernel(Module):
    r"""
    Creates a Laplace kernel
    """

    def __init__(self, loc, scale=1.0):
        super(LaplaceKernel, self).__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.abs().sum(-1)


class StudentTKernel(Module):
    r"""
    Creates a Student t kernel
    """

    def __init__(self, loc, scale=1.0):
        super(StudentTKernel, self).__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.pow(2).sum(-1).log1p()


class NormalKernel(Module):
    r"""
    Creates a normal (Gaussian) kernel
    """

    def __init__(self, loc, scale=1.0):
        super(NormalKernel, self).__init__()
        self.loc, self.scale = broadcast_all(loc, scale)

    def log_prob(self, value):
        y = (value - self.loc) / self.scale
        return -y.pow(2).div(2).sum(-1)


class VonMisesKernel(Module):
    """
    Spherical von Mises kernel
    """

    def __init__(self, concentration):
        self.concentration = concentration
        super().__init__()

    def log_prob(self, value):
        return -(value * self.concentration).sum(-1)


class CategoricalKernel(Module):
    """
    Categorical kernel
    """

    def __init__(self, logits):

        self.probs = logits.softmax(-1)
        super().__init__()

    def log_prob(self, value):
        return (self.probs * value.log_softmax(-1)).sum(-1)
