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

from selfsne.kernels import PAIRWISE_KERNELS
from selfsne.divergences import DIVERGENCES
from selfsne.baselines import BASELINES
from selfsne.utils import (
    remove_diagonal,
    off_diagonal,
    stop_gradient,
    random_sample_columns,
)

from typing import Optional, Union, Tuple


class DensityRatioEstimator(nn.Module):
    """
    A generalized multi-sample contrastive loss based on
    Noise Contrastive Estimation (NCE) that preserves local embedding
    structure by maximizing similarity for positive pairs and minimizing similarity
    for negative (noise) pairs.

    A user-selected similarity kernel is used to calculate the log density ratio ("logits")
    for each pair, which are then optimized by a user-selected divergence function to
    maximize logits for positive pairs (attraction) and minimize logits for negative pairs
    (repulsion).

    Args:
        kernel (Union[str, callable]): Similarity kernel used for calculating logits.
            Must be one of selfsne.kernels.KERNELS or a callable that takes in two 2D tensors and returns a pairwise 2D tensor.
            For example, "cauchy" can be used to produce t-SNE or UMAP embeddings, "normal" can be used to
            produce SNE embeddings, and "vonmises" can be used for (hyper)spherical embeddings.
        kernel_scale (Union[float, str]): Positive scale value for calculating logits.
            For loc-scale family kernels sqrt(embedding_dims) is recommended,
            which is calculated automatically when kernel_scale = "auto". Default is 1.0.
        temperature (float): The temperature for the logits. Larger values create more uniform embeddings. Default is 1.
        divergence (Union[str, callable]): Divergence function used for instance classification.
            Must be one of selfsne.divergences.DIVERGENCES or a callable that takes in two 2D tensors and returns a scalar.
        baseline (Union[str, float, callable]): The baseline for calculating the log density ratio.
            Must be a float, string (one of selfsne.baselines.BASELINES), or nn.Module such as from selfsne.baselines.
            Default is 0.
        num_negatives (Optional[int]): Number of negative samples to use. Default is None.
        embedding_decay (float): Weight decay for the embeddings. Default is 0.0.


    Returns:
        Tuple[torch.Tensor]: A tuple containing four tensors:
            [0] The mean of the positive logits, i.e., the diagonal entries of the
                kernel matrix (shape: (1,))
            [1] The mean of the negative logits, i.e., the off-diagonal entries of
                the kernel matrix (shape: (1,))
            [2] The mean of the log-baseline (shape: (1,))
            [3] The similarity loss, the combined attraction and repulsion terms and embedding decay (shape: (1,))

    References:
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
        [6] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
            A simple framework for contrastive learning of visual representations.
            In International conference on machine learning (pp. 1597-1607). PMLR.
        [7] Wang, M., & Wang, D. (2016). Vmf-sne: Embedding for spherical
            data. In 2016 IEEE International Conference on Acoustics, Speech
            and Signal Processing (ICASSP) (pp. 2344-2348). IEEE.

    """

    def __init__(
        self,
        kernel: Union[str, callable] = "cauchy",
        kernel_scale: Union[float, str] = 1.0,
        temperature: float = 1,
        divergence: Union[str, callable] = "kld",
        baseline: Union[str, float, callable] = 0,
        num_negatives: Optional[int] = None,
        embedding_decay: float = 0,
    ) -> None:

        super().__init__()
        if isinstance(kernel, str):
            self.kernel = PAIRWISE_KERNELS[kernel]
        else:
            self.kernel = kernel

        self.kernel_scale = kernel_scale

        if isinstance(divergence, str):
            self.divergence = DIVERGENCES[divergence]
        else:
            self.divergence = divergence

        if isinstance(baseline, str):
            self.baseline = BASELINES[baseline]()
        elif isinstance(baseline, (int, float)):
            self.baseline = BASELINES["constant"](baseline)
        else:
            self.baseline = baseline

        self.num_negatives = num_negatives
        self.embedding_decay = embedding_decay
        self.inverse_temperature = 1 / temperature

    def forward(
        self, z_x: torch.Tensor, z_y: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Computes similarity loss.

        Args:
            z_x (torch.Tensor): Input tensor with shape (batch_size, embedding_features)
                representing the features of the x domain.
            z_y (torch.Tensor): Input tensor with shape (batch_size, embedding_features)
                representing the embedding of the y labels.
            y (torch.Tensor, optional): Input tensor with shape (batch_size, **data_features) representing
                the original y data. Used when the baseline is a LearnedConditionalBaseline.
                Default: None.

        Returns:
            Tuple[torch.Tensor]: A tuple containing four tensors:
                [0] The mean of the positive logits, i.e., the diagonal entries of the
                    kernel matrix (shape: (1,))
                [1] The mean of the negative logits, i.e., the off-diagonal entries of
                    the kernel matrix (shape: (1,))
                [2] The mean of the log-baseline (shape: (1,))
                [3] The similarity loss, the combined attraction and repulsion terms and embedding decay (shape: (1,))
        """

        embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

        if self.kernel_scale == "auto":
            embedding_features = z_x.shape[1]
            self.kernel_scale = np.sqrt(embedding_features)

        logits = self.kernel(z_y, z_x, self.kernel_scale) * self.inverse_temperature
        pos_logits = diagonal(logits).unsqueeze(1)
        neg_logits = remove_diagonal(logits)
        if self.num_negatives:
            neg_logits = random_sample_columns(neg_logits, self.num_negatives)
        log_baseline = self.baseline(logits=neg_logits, y=y, z_y=z_y)
        pos_logits = pos_logits - log_baseline
        neg_logits = neg_logits - log_baseline
        attraction, repulsion = self.divergence(pos_logits, neg_logits)
        return (
            pos_logits.mean(),
            neg_logits.mean(),
            pos_logits.sigmoid().mean(),
            neg_logits.sigmoid().mean(),
            log_baseline.mean(),
            attraction.mean() + repulsion.mean() + embedding_decay,
        )


class RedundancyReduction(nn.Module):
    """
    Redundancy Reduction loss function [1] that creates an embedding by minimizing
    mean squared error between an identity matrix and the empirical cross-correlation matrix
    between positive pairs. This helps to maximize feature invariance to differences between
    positive pairs while minimizing redundancy (correlation) between features, preserving global
    structure in the embedding as a form of nonlinear Canonical Correlation Analysis (CCA) [2].

    Args:
        num_features (int): Number of embedding features.
        invariance_weight (float, optional): Weighting for the invariance term in the loss.
            Default: 1.0.
        redundancy_weight (float, optional): Weighting for the redundancy term in the loss.
            Default: 1.0.

    Returns:
        loss (torch.Tensor): The computed redundancy reduction loss.

    References:
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
        num_features: int = 2,
        invariance_weight: float = 1.0,
        redundancy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, affine=False)
        self.invariance_weight = invariance_weight
        self.redundancy_weight = redundancy_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.shape

        correlation = self.norm(x).T @ self.norm(y) / batch_size
        invariance = diagonal(correlation).add_(-1).pow_(2).mean()
        redundancy = off_diagonal(correlation).pow_(2).mean()

        return invariance * self.invariance_weight + redundancy * self.redundancy_weight


class VICReg(nn.Module):
    """
    Variance-Invariance-Covariance (VIC) regularization loss function [1] that creates an
    embedding by combining three terms:
    (1) a variance stabilizing term (hinge loss for feature-wise std. dev.),
    (2) an invariance term (mean squared error between positive pairs),
    (3) a covariance regularization term (minimize squared covariance) based on redundancy reduction [2].
    This helps to preserve global structure in the embedding as a form of Laplacian Eigenmaps [3].

    Args:
        num_features (int): Number of embedding features.
        eps (float, optional): A value added to the variance term for numerical stability. Default: 1e-8.
        variance_weight (float, optional): Weighting for the variance term in the loss. Default: 1.0.
        invariance_weight (float, optional): Weighting for the invariance term in the loss. Default: 1.0.
        covariance_weight (float, optional): Weighting for the covariance term in the loss. Default: 1.0.

    Returns:
        loss (torch.Tensor): The computed loss.

    References:
        [1] Bardes, A., Ponce, J., & LeCun, Y. (2021). VICReg:
        Variance-Invariance-Covariance Regularization for self-supervised
        learning. arXiv preprint arXiv:2105.04906.
        [2] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
        Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
        In International Conference on Machine Learning (pp. 12310-12320).
        PMLR.
        [3] Balestriero, R., & LeCun, Y. (2022). Contrastive and Non-Contrastive
        Self-Supervised Learning Recover Global and Local Spectral Embedding
        Methods. doi:10.48550/arxiv.2205.11508
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-8,
        variance_weight: float = 1.0,
        invariance_weight: float = 1.0,
        covariance_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.variance_weight = variance_weight
        self.invariance_weight = invariance_weight
        self.covariance_weight = covariance_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, embedding_features = x.shape
        invariance = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        std_x = diagonal(cov_x).add(self.eps).sqrt()
        std_y = diagonal(cov_y).add(self.eps).sqrt()
        variance = F.relu(1 - std_x).mean() / 2 + F.relu(1 - std_y).mean() / 2
        covariance = (
            off_diagonal(cov_x).pow_(2).sum().div(embedding_features) / 2
            + off_diagonal(cov_y).pow_(2).sum().div(embedding_features) / 2
        )

        return (
            variance * self.variance_weight
            + invariance * self.invariance_weight
            + covariance * self.covariance_weight
        )
