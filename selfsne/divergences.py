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
import torch.nn.functional as F

from selfsne.utils import logmeanexp


def categorical_cross_entropy(pos_logits, neg_logits):
    attraction = -pos_logits.mean()
    repulsion = logmeanexp(neg_logits)
    return attraction, repulsion


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


def interpolate_kullback_leibler_divergence(pos_logits, neg_logits, alpha=0.5):
    kld_attraction, kld_repulsion = kullback_leibler_divergence(
        pos_logits, neg_logits
    )
    rkld_attraction, rkld_repulsion = reverse_kullback_leibler_divergence(
        pos_logits, neg_logits
    )
    return (
        alpha * kld_attraction + (1 - alpha) * rkld_attraction,
        alpha * kld_repulsion + (1 - alpha) * rkld_repulsion,
    )


def jeffreys_divergence(pos_logits, neg_logits):
    return interpolate_kullback_leibler_divergence(pos_logits, neg_logits, 0.5)


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
    "categorical": categorical_cross_entropy,
    "binary": binary_cross_entropy,
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
