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

# Code for Barlow Twins redundancy reduction loss is adapted from:
# https://github.com/facebookresearch/barlowtwins/blob/main/main.py
# Code for VICReg loss is adapted from :
# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
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
import torch
from torch import nn
from torch import diagonal
import torch.nn.functional as F

from selfsne.kernels import KERNELS
from selfsne.divergences import DIVERGENCES
from selfsne.normalizers import NORMALIZERS
from selfsne.utils import remove_diagonal, off_diagonal, stop_gradient


class NCE(nn.Module):
    """
    A generalized multi-sample contrastive loss based on
    Noise Contrastive Estimation (NCE) [1] and its variants [2, 3]
    that preserves local embedding structure by maximizing similarity
    for positive pairs and minimizing similarity for negative (noise) pairs.

    A user-selected similarity kernel is used to calculate the log density ratio ("logits")
    for each pair, which are then optimized by a user-selected divergence function
    to maximize logits for positive pairs (attraction)
    and minimize logits for negative pairs (repulsion).

    Parameters
    ----------
    kernel: str
        Similarity kernel used for calculating logits.
        Must be one of selfsne.kernels.KERNELS.
        For example, "studentt" can be used to produce t-SNE [4] or UMAP [5]
        embeddings, "normal" can be used to produce SNE [6] embeddings,
        and "vonmises" can be used for (hyper)spherical embeddings [7, 8].

    kernel_scale: float or str, default=1.0
        Postive scale value for calculating logits.
        For loc-scale family kernels sqrt(embedding_dims) is recommended,
        which is calculated automatically when kernel_scale = "auto".

    divergence: str
        Divergence function used for instance classification.
        Must be one of selfsne.divergences.DIVERGENCES.
        For example, "categorical" applies categorical cross entropy, or InfoNCE [2],
        which can be used for t-SNE [4] and SimCLR [6] embeddings,
        while "binary" applies binary cross entropy, or NEG [4],
        which can be used for UMAP [5] embeddings.

    log_normalizer : float, str, or nn.Module, default = 0
        The log normalizer for calculating the log density ratio.
        Must be a float, one of selfsne.normalizers.NORMALIZERS,
        or nn.Module such as from selfsne.normalizers

    remove_diagonal : bool, default = True
        Whether to remove the positive logits (the diagonal of the negative logits)
        when calculating the repulsion term. The diagonal is removed when set to True.

    attraction_weight: float, default=1.0
        Weighting for the attraction term

    repulsion_weight: float, default=1.0
        Weighting for the repulsion term

    normalizer_weight: float, default=1.0
        Weighting for the normalizer term,
        where log_normalizer += log(normalizer_weight)

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

    [3] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
        Distributed representations of words and phrases and their compositionality.
        Advances in neural information processing systems, 26.

    [4] Van Der Maaten, L. (2009). Learning a parametric embedding
        by preserving local structure. In Artificial intelligence
        and statistics (pp. 384-391). PMLR.

    [5] Sainburg, T., McInnes, L., & Gentner, T. Q. (2021).
        Parametric UMAP Embeddings for Representation and Semisupervised
        Learning. Neural Computation, 33(11), 2881-2907.

    [6] Hinton, G. E., & Roweis, S. (2002). Stochastic neighbor embedding.
        Advances in neural information processing systems, 15.

    [7] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
        A simple framework for contrastive learning of visual representations.
        In International conference on machine learning (pp. 1597-1607). PMLR.

    [8] Wang, M., & Wang, D. (2016). Vmf-sne: Embedding for spherical
        data. In 2016 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) (pp. 2344-2348). IEEE.

    """

    def __init__(
        self,
        kernel="studentt",
        kernel_scale=1.0,
        divergence="categorical",
        log_normalizer=0.0,
        remove_diagonal=True,
        attraction_weight=1.0,
        repulsion_weight=1.0,
        normalizer_weight=1.0,
    ):
        super().__init__()
        if isinstance(kernel, str):
            self.kernel = KERNELS[kernel]
        else:
            self.kernel = kernel

        self.kernel_scale = kernel_scale

        if isinstance(divergence, str):
            self.divergence = DIVERGENCES[divergence]
        else:
            self.divergence = divergence

        if isinstance(log_normalizer, str):
            self.log_normalizer = NORMALIZERS[log_normalizer]()
        elif isinstance(log_normalizer, (int, float)):
            self.log_normalizer = NORMALIZERS["constant"](log_normalizer)
        else:
            self.log_normalizer = log_normalizer

        self.remove_diagonal = remove_diagonal
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.log_normalizer_weight = np.log(normalizer_weight)

    def forward(self, x, y):
        if self.kernel_scale == "auto":
            self.kernel_scale = np.sqrt(x.shape[1])

        logits = self.kernel(x, y.unsqueeze(1), self.kernel_scale)
        log_normalizer = self.log_normalizer(y, logits)
        logits = logits - (log_normalizer + self.log_normalizer_weight)

        pos_logits = diagonal(logits)
        neg_logits = remove_diagonal(logits) if self.remove_diagonal else logits
        attraction, repulsion = self.divergence(pos_logits, neg_logits)
        return (
            attraction.mean() * self.attraction_weight
            + repulsion.mean() * self.repulsion_weight
        )


class RedundancyReduction(nn.Module):
    """Redundancy Reduction [1]

    A self-supervised loss that creates an an embedding by minimizing
    mean squared error between an identity matrix and the
    empirical cross-correlation matrix between positive pairs,
    which maximizes feature invariance to differences between positive pairs
    while also minimizing redundancy (correlation) between features.
    This helps to preserve global structure in the embedding as a form of
    nonlinear Canonical Correlation Analysis (CCA) [2].

    Parameters
    ----------
    num_features: int
        Number of embedding features

    invariance_weight: float, default=1.0
        Weighting for the invariance term

    redundancy_weight: float, default=1.0
        Weighting for the redundancy term


    References
    ----------
    [1] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
        Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
        In International Conference on Machine Learning (pp. 12310-12320).
        PMLR.

    [2] Balestriero, R., & LeCun, Y. (2022). Contrastive and Non-Contrastive
        Self-Supervised Learning Recover Global and Local Spectral Embedding
        Methods. doi:10.48550/arxiv.2205.11508

    """

    def __init__(self, num_features=2, invariance_weight=1.0, redundancy_weight=1.0):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(num_features, affine=False)
        self.invariance_weight = invariance_weight
        self.redundancy_weight = redundancy_weight

    def forward(self, x, y):
        n, d = x.shape

        correlation = self.normalizer(x).T @ self.normalizer(y) / n
        invariance = diagonal(correlation).add_(-1).pow_(2).mean()  # .sum() / d
        redundancy = off_diagonal(correlation).pow_(2).mean()  # .sum() / (d * d - d)

        return invariance * self.invariance_weight + redundancy * self.redundancy_weight


class VICReg(nn.Module):
    """Variance-Invariance-Covariance Regularization [1]

    A self-supervised embedding loss that combinines three terms:
    (1) a variance stabilizing term (hinge loss for feature-wise std. dev.),
    (2) an invariance term (mean squared error between positive pairs),
    (3) a covariance regularization (minimize squared covariance), which is
    a decorrelation mechanism based on redundancy reduction [2].
    This helps to preserve global structure in the embedding as a form of
    Laplacian Eigenmaps [3].

    Parameters
    ----------
    num_features: int
        Number of embedding features

    eps: float, default=1e-8
        A value added to the variance term for numerical stability

    variance_weight: float, default=1.0
        Weighting for the variance term

    invariance_weight: float, default=1.0
        Weighting for the invariance term

    covariance_weight: float, default=1.0
        Weighting for the covariance term

    References
    ----------
    [1] Bardes, A., Ponce, J., & LeCun, Y. (2021). VICReg:
        Variance-Invariance-Covariance Regularization for self-supervised
        learning. arXiv preprint arXiv:2105.04906.

    [1] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
        Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
        In International Conference on Machine Learning (pp. 12310-12320).
        PMLR.

    [2] Balestriero, R., & LeCun, Y. (2022). Contrastive and Non-Contrastive
        Self-Supervised Learning Recover Global and Local Spectral Embedding
        Methods. doi:10.48550/arxiv.2205.11508

    """

    def __init__(
        self,
        num_features,
        eps=1e-8,
        variance_weight=1.0,
        invariance_weight=1.0,
        covariance_weight=1.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.variance_weight = variance_weight
        self.invariance_weight = invariance_weight
        self.covariance_weight = covariance_weight

    def forward(self, x, y):

        n, d = x.shape
        invariance = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        cov_x = (x.T @ x) / (n - 1)
        cov_y = (y.T @ y) / (n - 1)
        std_x = diagonal(cov_x).add(self.eps).sqrt()
        std_y = diagonal(cov_y).add(self.eps).sqrt()
        variance = F.relu(1 - std_x).mean() / 2 + F.relu(1 - std_y).mean() / 2
        covariance = (
            off_diagonal(cov_x).pow_(2).sum().div(d) / 2
            + off_diagonal(cov_y).pow_(2).sum().div(d) / 2
        )

        return (
            variance * self.variance_weight
            + invariance * self.invariance_weight
            + covariance * self.covariance_weight
        )
