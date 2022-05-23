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


def categorical_cross_entropy(pos_logits, neg_logits):
    attract = -pos_logits
    repel = logmeanexp(neg_logits)
    return attract + repel


def binary_cross_entropy(pos_logits, neg_logits):
    attract = F.softplus(-pos_logits)
    repel = F.softplus(neg_logits).mean(-1)
    return attract + repel


def cauchy_schwarz_divergence(pos_logits, neg_logits):
    attract = -pos_logits
    repel = 0.5 * logmeanexp(2 * neg_logits)
    return attract + repel


# discriminators from Qin et al. (2020) https://arxiv.org/abs/1811.09567
def wasserstein_logits(pos_logits, neg_logits):
    attract = -pos_logits
    repel = neg_logits.mean(-1)
    return attract + repel


def wasserstein(pos_logits, neg_logits):
    attract = -pos_logits.exp()
    repel = logmeanexp(neg_logits).exp()
    return attract + repel


def least_squares(pos_logits, neg_logits):
    attract = pos_logits.expm1().pow(2)
    repel = logmeanexp(neg_logits.mul(2)).exp()
    return attract + repel


def zero_centered_least_squares(pos_logits, neg_logits):
    attract = pos_logits.expm1().pow(2)
    repel = logmeanexp(F.softplus(neg_logits).mul(2)).exp()
    return attract + repel


def cosine(pos_logits, neg_logits):
    attract = -pos_logits.expm1().cos()
    repel = -neg_logits.exp().cos().add(1).mean(-1)
    return attract + repel


# discriminators from Nowozin et al. (2016) https://arxiv.org/abs/1606.00709
def jensen_shannon_divergence(pos_logits, neg_logits):
    def activation(v):
        return np.log(2) + F.logsigmoid(v)

    def conjugate(t):
        return -(2 - t.exp()).log()

    attract = -activation(pos_logits)
    repel = conjugate(activation(neg_logits)).mean(-1)
    return attract + repel


def kullback_leibler_divergence(pos_logits, neg_logits):
    def activation(v):
        return v

    def conjugate(t):
        return (t - 1).exp()

    attract = -activation(pos_logits)
    repel = conjugate(activation(neg_logits)).mean(-1)
    return attract + repel


def reverse_kullback_leibler_divergence(pos_logits, neg_logits):
    def activation(v):
        return -torch.exp(-v)

    def conjugate(t):
        return -1 - torch.log(-t)

    attract = -activation(pos_logits)
    repel = conjugate(activation(neg_logits)).mean(-1)
    return attract + repel


def pearson_chi_sq(pos_logits, neg_logits):
    def activation(v):
        return v

    def conjugate(t):
        return 0.25 * t ** 2 + t

    attract = -activation(pos_logits)
    repel = conjugate(activation(neg_logits)).mean(-1)
    return attract + repel


def squared_hellinger(pos_logits, neg_logits):
    def activation(v):
        return 1 - (-v).exp()

    def conjugate(t):
        return t / (1 - t)

    attract = -activation(pos_logits)
    repel = conjugate(activation(neg_logits)).mean(-1)
    return attract + repel


DISCRIMINATORS = {
    "categorical": categorical_cross_entropy,
    "binary": binary_cross_entropy,
    "bernoulli": binary_cross_entropy,
    "jsd": jensen_shannon_divergence,
    "reverse_kld": reverse_kullback_leibler_divergence,
    "kld": kullback_leibler_divergence,
    "pearson": pearson_chi_sq,
    "squared_hellinger": squared_hellinger,
    "wasserstein": wasserstein,
    "wasserstein_logits": wasserstein_logits,
    "least_squares": least_squares,
    "centered_least_squares": zero_centered_least_squares,
    "cosine": cosine,
    "cauchy_schwarz": cauchy_schwarz_divergence,
}
