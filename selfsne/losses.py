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

# Code for Barlow's Twins redundancy reduction loss is adapted from:
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
from selfsne.discriminators import DISCRIMINATORS


def query_logits(query, pos_key, neg_key, kernel, kernel_scale=1.0):
    pos_logits = kernel(query, kernel_scale).log_prob(pos_key)
    neg_logits = kernel(query.unsqueeze(1), kernel_scale).log_prob(neg_key)
    return pos_logits, neg_logits


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


class NCE(nn.Module):
    """Noise Contrastive Estimation [1, 2]

    A generalized multi-sample contrastive loss that preserves local embedding
    structure by maximizing similarity for positive pairs and minimizing
    similarity for negative (noise) pairs.

    A user-selected similarity kernel is used to calculate logits for each
    pair, which are then optimized by a user-selected discriminator function
    to maximize logits for positive pairs (attractive forces)
    and minimize logits for negative pairs (repulsive forces).

    Parameters
    ----------
    kernel: str
        Similarity kernel used for calculating discriminator logits.
        Must be one of selfsne.kernels.KERNELS.
        For example, "studentt" can be used to produce t-SNE [3] or UMAP [4]
        embeddings, "normal" can be used to produce SNE [5] embeddings,
        and "vonmises" can be used for (hyper)spherical embeddings [6, 7].

    discriminator: str
        Discriminator function used for instance classification.
        Must be one of selfsne.discriminators.DISCRIMINATORS.
        For example, "categorical" applies categorical cross entropy,
        or InfoNCE [2], which can be used for t-SNE [3] and SimCLR [6]
        embeddings, while "binary" applies binary cross entropy,
        or classic NCE [1], which can be used for UMAP [4] embeddings.

    kernel_scale: float, default=1.0
        Postive scale value for calculating logits.
        For loc-scale family kernels sqrt(embedding_dims) is recommended.

    References
    ----------
    [1] Gutmann, M., & Hyvärinen, A. (2010). Noise-contrastive estimation:
        A new estimation principle for unnormalized statistical models.
        In Proceedings of the thirteenth international conference on artificial
        intelligence and statistics (pp. 297-304). JMLR Workshop and
        Conference Proceedings.

    [2] Oord, A. V. D., Li, Y., & Vinyals, O. (2018).
        Representation learning with contrastive predictive coding.
        arXiv preprint arXiv:1807.03748.

    [3] Van Der Maaten, L. (2009). Learning a parametric embedding
        by preserving local structure. In Artificial intelligence
        and statistics (pp. 384-391). PMLR.

    [4] Sainburg, T., McInnes, L., & Gentner, T. Q. (2021).
        Parametric UMAP Embeddings for Representation and Semisupervised
        Learning. Neural Computation, 33(11), 2881-2907.

    [5] Hinton, G. E., & Roweis, S. (2002). Stochastic neighbor embedding.
        Advances in neural information processing systems, 15.

    [6] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November).
        A simple framework for contrastive learning of visual representations.
        In International conference on machine learning (pp. 1597-1607). PMLR.

    [7] Wang, M., & Wang, D. (2016, March). Vmf-sne: Embedding for spherical
        data. In 2016 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) (pp. 2344-2348). IEEE.

    """

    def __init__(
        self, kernel="studentt", discriminator="categorical", kernel_scale=1.0
    ):
        super().__init__()
        self.kernel = KERNELS[kernel]
        self.discriminator = DISCRIMINATORS[discriminator]
        self.kernel_scale = kernel_scale

    def forward(self, query, pos_key, neg_key=None):
        pos_logits, neg_logits = query_logits(
            query,
            pos_key,
            pos_key if neg_key is None else neg_key,
            self.kernel,
            self.kernel_scale,
        )
        return self.discriminator(pos_logits, neg_logits)


class RedundancyReduction(nn.Module):
    """Redundancy Reduction [1]

    A self-supervised loss that reduces feature redundancy for an embedding
    by minimizing mean squared error between an identity matrix and the
    empirical cross-correlation matrix between positive pairs.
    This helps to preserve global structure in the embedding as a form of
    nonlinear canonical correlation analysis (CCA) [2].

    Parameters
    ----------
    num_features: int
        Number of embedding features

    References
    ----------
    [1] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
        Barlow twins: Self-supervised learning via redundancy reduction.
        In International Conference on Machine Learning (pp. 12310-12320).
        PMLR.

    [2] Balestriero, R., & LeCun, Y. (2022). Contrastive and Non-Contrastive
        Self-Supervised Learning Recover Global and Local Spectral Embedding
        Methods. doi:10.48550/arxiv.2205.11508

    """

    def __init__(self, num_features=2):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, query, key):
        return redundancy_reduction(query, key, self.normalizer)
