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
from scipy.special import logit, expit

import torch
import torch.nn as nn
import torch.nn.functional as F

from selfsne.utils import logmeanexp, straight_through_estimator


def density_ratio(pos_logits, neg_logits):
    attraction = -pos_logits.mean()
    repulsion = torch.zeros_like(attraction)
    return attraction, repulsion


def total_variation_distance(pos_logits, neg_logits):
    f_prime_pos = straight_through_estimator(
        pos_logits, (pos_logits > 0) * 0.5 - (pos_logits <= 0) * 0.5
    )
    f_prime_neg = straight_through_estimator(
        neg_logits, (neg_logits > 0) * 0.5 - (neg_logits <= 0) * 0.5
    )

    attraction = -f_prime_pos.mean()
    repulsion = (neg_logits.exp() * f_prime_neg - 0.5 * neg_logits.expm1().abs()).mean()

    return attraction, repulsion


def wasserstein(pos_logits, neg_logits):
    attraction = -pos_logits.mean()
    repulsion = neg_logits.mean()
    return attraction, repulsion


class BaseNCE(nn.Module):
    def __init__(self, alpha=0.5, k=None, log_k=None):
        super(BaseNCE, self).__init__()

        assert (k is None) or (log_k is None), "Cannot define both k and log_k"

        if log_k is not None:
            self.log_k = log_k
            self.k = np.exp(self.log_k)
            self.alpha_logit = self.log_k
            self.alpha = expit(self.log_k)
        elif k is not None:
            self.k = k
            self.log_k = np.log(self.k)
            self.alpha_logit = self.log_k
            self.alpha = expit(self.log_k)
        else:
            self.alpha = alpha
            self.alpha_logit = logit(alpha)
            self.log_k = self.alpha_logit
            self.k = np.exp(self.log_k)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha:.1e}, k={self.k:.1e}, log_k={self.log_k:.1e})"

    def forward(self):
        raise NotImplementedError


class ForwardNCE(BaseNCE):
    def forward(self, pos_logits, neg_logits):
        attraction = -F.logsigmoid(pos_logits - self.log_k).mean()
        repulsion = -self.k * F.logsigmoid(-neg_logits + self.log_k).mean()
        return attraction, repulsion


class ReverseNCE(BaseNCE):
    def forward(self, pos_logits, neg_logits):
        attraction = -self.k * F.logsigmoid(pos_logits + self.log_k).mean()
        repulsion = -F.logsigmoid(-neg_logits - self.log_k).mean()
        return attraction, repulsion


class AlphaNCE(BaseNCE):
    def __init__(self, alpha=0.5, k=None, log_k=None):
        super(AlphaNCE, self).__init__(alpha=alpha, k=k, log_k=log_k)
        if self.alpha == 0:
            self.divergence = kullback_leibler_divergence
        elif self.alpha == 1:
            self.divergence = reverse_kullback_leibler_divergence
        elif self.alpha == 0.5:
            self.divergence = binary_cross_entropy
        elif self.alpha < 0.5:
            self.divergence = ForwardNCE(alpha=1 - self.alpha)
        elif self.alpha > 0.5:
            self.divergence = ReverseNCE(alpha=self.alpha)

    def forward(self, pos_logits, neg_logits):
        return self.divergence(pos_logits, neg_logits)


def binary_cross_entropy(pos_logits, neg_logits):
    attraction = -F.logsigmoid(pos_logits).mean()
    repulsion = -F.logsigmoid(-neg_logits).mean()
    return attraction, repulsion


def jensen_shannon_divergence(pos_logits, neg_logits):
    attraction, repulsion = binary_cross_entropy(pos_logits, neg_logits)
    attraction = 2 * attraction - np.log(4)
    repulsion = 2 * repulsion - np.log(4)
    return attraction, repulsion


def kullback_leibler_divergence(pos_logits, neg_logits):
    attraction = -pos_logits.mean()
    repulsion = logmeanexp(neg_logits).expm1()
    return attraction, repulsion


def reverse_kullback_leibler_divergence(pos_logits, neg_logits):
    attraction = logmeanexp(-pos_logits).expm1()
    repulsion = neg_logits.mean()
    return attraction, repulsion


class InterpolateDivergences(nn.Module):
    def __init__(self, divergence_a, divergence_b, alpha=0.5):
        super(InterpolateDivergences, self).__init__()
        self.divergence_a = divergence_a
        self.divergence_b = divergence_b
        self.alpha = alpha

    def forward(self, pos_logits, neg_logits):
        attraction_a, repulsion_a = self.divergence_a(pos_logits, neg_logits)
        attraction_b, repulsion_b = self.divergence_b(pos_logits, neg_logits)
        return (
            torch.lerp(attraction_a, attraction_b, self.alpha),
            torch.lerp(repulsion_a, repulsion_b, self.alpha),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha:.3f})"


jeffreys_divergence = InterpolateDivergences(
    kullback_leibler_divergence, reverse_kullback_leibler_divergence, alpha=0.5
)


def squared_hellinger(pos_logits, neg_logits):
    attraction = 2 * logmeanexp(-pos_logits * 0.5).expm1()
    repulsion = 2 * logmeanexp(neg_logits * 0.5).expm1()
    return attraction, repulsion


def pearson_chi_sq(pos_logits, neg_logits):
    attraction = -logmeanexp(pos_logits).expm1()
    repulsion = logmeanexp(2 * neg_logits).expm1() / 2
    return attraction, repulsion


def squared_le_cam_distance(pos_logits, neg_logits):
    attraction = logmeanexp(2 * F.logsigmoid(-pos_logits)).add(np.log(4)).expm1()
    repulsion = logmeanexp(2 * F.logsigmoid(neg_logits)).add(np.log(4)).expm1()
    return attraction, repulsion


def neymann_divergence(pos_logits, neg_logits):
    attraction = logmeanexp(-2 * pos_logits).add(-np.log(2)).exp() - 0.5
    repulsion = 1 - logmeanexp(-neg_logits).exp()
    return attraction, repulsion


def softened_reverse_kullback_leibler_divergence(pos_logits, neg_logits):
    attraction = (
        logmeanexp(-pos_logits).add(np.log(2)).exp()
        + 2 * F.logsigmoid(pos_logits).mean()
        + 2
        + np.log(4)
    )
    repulsion = 2 * F.logsigmoid(neg_logits).mean() + np.log(4)
    return attraction, repulsion


DIVERGENCES = {
    "dr": density_ratio,
    "wasserstein": wasserstein,
    "binary": binary_cross_entropy,
    "bce": binary_cross_entropy,
    "jsd": jensen_shannon_divergence,
    "rkld": reverse_kullback_leibler_divergence,
    "kld": kullback_leibler_divergence,
    "pearson": pearson_chi_sq,
    "hellinger": squared_hellinger,
    "jeffreys": jeffreys_divergence,
    "le_cam": squared_le_cam_distance,
    "neymann": neymann_divergence,
    "soft_rkld": softened_reverse_kullback_leibler_divergence,
}
