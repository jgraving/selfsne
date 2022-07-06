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

import torch
from torch import nn
import torch.nn.functional as F

from selfsne.utils import logmeanexp
from selfsne.normalizers import LogEMA


class MINE(nn.Module):
    """
    MINE estimator with debiased gradients
    from Belghazi et al. (2021) https://arxiv.org/abs/1801.04062
    """

    def __init__(self, momentum=0.99):
        super().__init__()
        self.log_ema = LogEMA(momentum, gradient=True)

    def forward(self, pos_logits, neg_logits):
        attraction = -pos_logits
        repulsion = self.log_ema(neg_logits)
        return attraction, repulsion


def noise_contrastive_estimation(pos_logits, neg_logits):
    noise_log_prob = np.log(neg_logits.shape[1]) + torch.zeros(
        1, device=neg_logits.device
    )
    attraction = torch.logaddexp(pos_logits, noise_log_prob) - pos_logits
    repulsion = (torch.logaddexp(neg_logits, noise_log_prob) - noise_log_prob).mean(-1)
    return attraction, repulsion


def categorical_cross_entropy(pos_logits, neg_logits):
    attraction = -pos_logits
    repulsion = logmeanexp(neg_logits, dim=-1)
    return attraction, repulsion


def joint_categorical_cross_entropy(pos_logits, neg_logits):
    attraction = -pos_logits
    repulsion = logmeanexp(neg_logits)
    return attraction, repulsion


def binary_cross_entropy(pos_logits, neg_logits):
    attraction = -F.logsigmoid(pos_logits)
    # use numerically stable repulsion term
    # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
    # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
    repulsion = -(F.logsigmoid(neg_logits) - neg_logits).mean(-1)
    return attraction, repulsion


def cauchy_schwarz_divergence(pos_logits, neg_logits):
    attraction = -pos_logits
    repulsion = 0.5 * logmeanexp(2 * neg_logits, dim=-1)
    return attraction, repulsion


# discriminators from Poole et al. (2019) https://arxiv.org/abs/1905.06922
def tuba(pos_logits, neg_logits):
    attraction = -pos_logits
    repulsion = logmeanexp(neg_logits).expm1()
    return attraction, repulsion


def nwj(pos_logits, neg_logits):
    return tuba(pos_logits - 1, neg_logits - 1)


# discriminators from Qin et al. (2020) https://arxiv.org/abs/1811.09567
def wasserstein_logits(pos_logits, neg_logits):
    attraction = -pos_logits
    repulsion = neg_logits.mean(-1)
    return attraction, repulsion


def wasserstein(pos_logits, neg_logits):
    attraction = -pos_logits.exp()
    repulsion = logmeanexp(neg_logits, dim=-1).exp()
    return attraction, repulsion


def least_squares(pos_logits, neg_logits):
    attraction = pos_logits.expm1().pow(2)
    repulsion = logmeanexp(neg_logits.mul(2), dim=-1).exp()
    return attraction, repulsion


def zero_centered_least_squares(pos_logits, neg_logits):
    attraction = pos_logits.expm1().pow(2)
    repulsion = logmeanexp(F.softplus(neg_logits).mul(2), dim=-1).exp()
    return attraction, repulsion


def cosine(pos_logits, neg_logits):
    attraction = -pos_logits.expm1().cos()
    repulsion = -neg_logits.exp().cos().add(1).mean(-1)
    return attraction, repulsion


# discriminators from Nowozin et al. (2016) https://arxiv.org/abs/1606.00709
def jensen_shannon_divergence(pos_logits, neg_logits):
    def activation(v):
        return np.log(2) + F.logsigmoid(v)

    def conjugate(t):
        return -(2 - t.exp()).log()

    attraction = -activation(pos_logits)
    repulsion = conjugate(activation(neg_logits)).mean(-1)
    return attraction, repulsion


def kullback_leibler_divergence(pos_logits, neg_logits):
    def activation(v):
        return v

    def conjugate(t):
        return (t - 1).exp()

    attraction = -activation(pos_logits)
    repulsion = conjugate(activation(neg_logits)).mean(-1)
    return attraction, repulsion


def reverse_kullback_leibler_divergence(pos_logits, neg_logits):
    def activation(v):
        return -torch.exp(-v)

    def conjugate(t):
        return -1 - torch.log(-t)

    attraction = -activation(pos_logits)
    repulsion = conjugate(activation(neg_logits)).mean(-1)
    return attraction, repulsion


def pearson_chi_sq(pos_logits, neg_logits):
    def activation(v):
        return v

    def conjugate(t):
        return 0.25 * t ** 2 + t

    attraction = -activation(pos_logits)
    repulsion = conjugate(activation(neg_logits)).mean(-1)
    return attraction, repulsion


def squared_hellinger(pos_logits, neg_logits):
    def activation(v):
        return 1 - (-v).exp()

    def conjugate(t):
        return t / (1 - t)

    attraction = -activation(pos_logits)
    repulsion = conjugate(activation(neg_logits)).mean(-1)
    return attraction, repulsion


DISCRIMINATORS = {
    "nce": noise_contrastive_estimation,
    "mine": MINE(),
    "categorical": categorical_cross_entropy,
    "infonce": categorical_cross_entropy,
    "joint_categorical": joint_categorical_cross_entropy,
    "tuba": tuba,
    "nwj": nwj,
    "binary": binary_cross_entropy,
    "neg": binary_cross_entropy,
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
