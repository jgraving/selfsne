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

import numpy as np


def infonce(z_a, z_b, kernel):
    n = z_a.shape[0]
    kernel = kernel(z_a)
    conditional = kernel.log_prob(z_b)
    marginal = kernel.log_prob(z_b.unsqueeze(1)).logsumexp(1) - np.log(n)
    return -conditional + marginal


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    # source: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def redundancy_reduction(z_a, z_b, normalizer):
    n = z_a.shape[0]
    # d = z_a.shape[1]

    c_z = normalizer(z_a).T @ normalizer(z_b) / n
    invariance = (1 - c_z.diagonal()).pow(2).mean()  # .sum() / d
    redundancy = off_diagonal(c_z).pow(2).mean()  # .sum() / (d * d - d)

    return invariance + redundancy
