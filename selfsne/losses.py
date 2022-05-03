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

# Code for redundancy reduction loss is adapted from:
# https://github.com/facebookresearch/barlowtwins/blob/main/main.py
# Under the following License:

# MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import diagonal

from selfsne.kernels import KERNELS

def logmeanexp(x, dim=-1):
    return x.logsumexp(dim) - np.log(x.shape[dim])


def query_logits(query, pos_key, neg_key, kernel):
    pos_logits = kernel(query).log_prob(pos_key)
    neg_logits = kernel(query.unsqueeze(1)).log_prob(neg_key)
    return pos_logits, neg_logits

def categorical_cross_entropy(pos_logits, neg_logits):
    attract = -pos_logits
    repel = logmeanexp(neg_logits)
    return attract + repel


def binary_cross_entropy(pos_logits, neg_logits):
    attract = -F.logsigmoid(pos_logits)
    # use numerically stable repulsion term
    # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
    # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
    repel = -(F.logsigmoid(neg_logits) - neg_logits).mean(-1)
    return attract + repel


CE_LOSSES = {
    "categorical": categorical_cross_entropy,
    "binary": binary_cross_entropy,
    "bernoulli": binary_cross_entropy,
}


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def redundancy_reduction(query, key, normalizer):
    n, d = query.shape

    correlation = normalizer(query).T @ normalizer(key) / n
    invariance = diagonal(correlation).add_(-1).pow_(2).mean()  # .sum() / d
    redundancy = off_diagonal(correlation).pow_(2).mean()  # .sum() / (d * d - d)

    return invariance + redundancy


class InfoNCE(nn.Module):
    def __init__(self, kernel="studentt", cross_entropy="categorical"):
        super().__init__()
        self.kernel = KERNELS[kernel]
        self.cross_entropy = CE_LOSSES[cross_entropy]

    def forward(self, query, pos_key, neg_key=None):
        pos_logits, neg_logits = query_logits(
            query, pos_key, pos_key if neg_key is None else neg_key, self.kernel
        )
        return self.cross_entropy(pos_logits, neg_logits)


class RedundancyReduction(nn.Module):
    def __init__(self, num_features=2):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, query, key):
        return redundancy_reduction(query, key, self.normalizer)
