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
import torch.nn.functional as F


def logmeanexp(x, dim=-1):
    return x.logsumexp(dim) - np.log(x.shape[dim])


def categorical_infonce(query, pos_key, neg_key, kernel):

    pos_logits = kernel(query).log_prob(pos_key)
    neg_logits = kernel(query.unsqueeze(1)).log_prob(neg_key)

    attract = -pos_logits
    repel = logmeanexp(neg_logits)

    return attract + repel


def binary_infonce(query, pos_key, neg_key, kernel):

    pos_logits = kernel(query).log_prob(pos_key).unsqueeze(-1)
    neg_logits = kernel(query.unsqueeze(1)).log_prob(neg_key)

    attract = -F.log_sigmoid(pos_logits)

    # use numerically stable repulsion term
    # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
    # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
    repel = -(F.logsigmoid(neg_logits) - neg_logits).mean(-1)

    return attract + repel


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    # source: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def redundancy_reduction(query, key, normalizer):
    n, d = z_a.shape

    correlation = normalizer(query).T @ normalizer(key) / n
    invariance = (1 - correlation.diagonal()).pow(2).mean()  # .sum() / d
    redundancy = off_diagonal(correlation).pow(2).mean()  # .sum() / (d * d - d)

    return invariance + redundancy
