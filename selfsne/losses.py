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
from torch.nn import ModuleList
import torch.nn.functional as F

from torchmetrics.functional import accuracy, precision, recall

from selfsne.kernels import PAIRWISE_KERNELS
from selfsne.divergences import DIVERGENCES
from selfsne.baselines import BASELINES
from selfsne.utils import (
    remove_diagonal,
    off_diagonal,
    stop_gradient,
    random_sample_columns,
)

from typing import Optional, Union, Tuple, List, Callable
from copy import deepcopy


def classifier_metrics(pos_logits, neg_logits):
    """
    Calculates accuracy, recall, and precision given positive and negative logits.

    Args:
        pos_logits (torch.Tensor): A tensor of positive logits of shape (batch_size,).
        neg_logits (torch.Tensor): A tensor of negative logits of shape (batch_size,).
        threshold (float): A threshold to convert logits to binary predictions. Defaults to 0.5.

    Returns:
        accuracy (float): The accuracy score
        recall (float): The recall score.
        precision (float): The precision score.
    """
    # Combine positive and negative logits
    logits = torch.cat((pos_logits, neg_logits), dim=0)

    # Convert logits to probabilities using sigmoid function
    preds = torch.sigmoid(logits)

    # Create binary labels (1 for positive, 0 for negative)
    target = torch.cat(
        (
            torch.ones_like(pos_logits, dtype=torch.long, device=preds.device),
            torch.zeros_like(neg_logits, dtype=torch.long, device=preds.device),
        ),
        dim=0,
    )
    return accuracy(preds, target), recall(preds, target), precision(preds, target)


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
            Default is "batch".
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
        baseline: Union[str, float, callable] = "batch",
        num_negatives: Optional[int] = None,
        embedding_decay: float = 0,
        symmetric_negatives: bool = False,
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
        self.symmetric_negatives = symmetric_negatives

    def forward(
        self,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
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
            Tuple[torch.Tensor]: A tuple containing six tensors:
                [0] The mean of the positive logits, i.e., the diagonal entries of the
                    kernel matrix (shape: (1,))
                [1] The mean of the negative logits, i.e., the off-diagonal entries of
                    the kernel matrix (shape: (1,))
                [2] The mean of the positive probs, i.e., the diagonal entries of the
                    kernel matrix (shape: (1,))
                [3] The mean of the negative probs, i.e., the off-diagonal entries of
                    the kernel matrix (shape: (1,))
                [4] The mean of the log-baseline (shape: (1,))
                [5] The similarity loss, the combined attraction and repulsion terms and embedding decay (shape: (1,))
        """

        embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

        if self.kernel_scale == "auto":
            embedding_features = z_x.shape[1]
            self.kernel_scale = np.sqrt(embedding_features)

        logits = self.kernel(z_y, z_x, self.kernel_scale) * self.inverse_temperature
        pos_logits = diagonal(logits).unsqueeze(1)
        neg_logits = remove_diagonal(logits)
        if self.symmetric_negatives:
            logits = self.kernel(z_y, z_y, self.kernel_scale) * self.inverse_temperature
            neg_logits = torch.cat([neg_logits, remove_diagonal(logits)], dim=-1)
        if self.num_negatives:
            neg_logits = random_sample_columns(neg_logits, self.num_negatives)
        log_baseline = self.baseline(logits=neg_logits, y=y, z_y=z_y)
        pos_logits = pos_logits - log_baseline
        neg_logits = neg_logits - log_baseline
        attraction, repulsion = self.divergence(pos_logits, neg_logits)
        with torch.no_grad():
            accuracy, recall, precision = classifier_metrics(
                pos_logits.flatten(), neg_logits.flatten()
            )
        return (
            pos_logits.mean(),
            neg_logits.mean(),
            pos_logits.sigmoid().mean(),
            neg_logits.sigmoid().mean(),
            accuracy,
            recall,
            precision,
            log_baseline.mean(),
            attraction.mean() + repulsion.mean() + embedding_decay,
        )


class MultiHeadDensityRatioEstimator(nn.Module):
    def __init__(
        self,
        num_heads: int,
        kernels: Union[str, Callable, List[Union[str, Callable]]] = "cauchy",
        kernel_scales: Union[float, str, List[Union[float, str]]] = 1.0,
        temperatures: Union[float, List[float]] = 1,
        divergences: Union[str, Callable, List[Union[str, Callable]]] = "kld",
        baselines: Union[
            str, float, Callable, List[Union[str, float, Callable]]
        ] = "batch",
        num_negatives: Optional[int] = None,
        embedding_decay: float = 0,
        symmetric_negatives: bool = False,
    ) -> None:

        super().__init__()

        if isinstance(kernels, list):
            assert (
                len(kernels) == num_heads
            ), "Length of kernels list should match num_heads"
            self.kernels = [
                PAIRWISE_KERNELS[k] if isinstance(k, str) else k for k in kernels
            ]
        else:
            kernels = PAIRWISE_KERNELS[kernels] if isinstance(kernels, str) else kernels
            self.kernels = [kernels] * num_heads

        if isinstance(kernel_scales, list):
            assert (
                len(kernel_scales) == num_heads
            ), "Length of kernel_scales list should match num_heads"
            self.kernel_scales = kernel_scales
        else:
            self.kernel_scales = [kernel_scales] * num_heads

        if isinstance(temperatures, list):
            assert (
                len(temperatures) == num_heads
            ), "Length of temperatures list should match num_heads"
            self.inverse_temperatures = [1 / t for t in temperatures]
        else:
            self.inverse_temperatures = [1 / temperatures] * num_heads

        if isinstance(divergences, list):
            assert (
                len(divergences) == num_heads
            ), "Length of divergences list should match num_heads"
            self.divergences = [
                DIVERGENCES[d] if isinstance(d, str) else d for d in divergences
            ]
        else:
            divergence = (
                DIVERGENCES[divergences]
                if isinstance(divergences, str)
                else divergences
            )
            self.divergences = [divergence] * num_heads

        if isinstance(baselines, list):
            assert (
                len(baselines) == num_heads
            ), "Length of baselines list should match num_heads"
            self.baselines = ModuleList()
            for b in baselines:
                if isinstance(b, str):
                    self.baselines.append(BASELINES[b]())
                elif isinstance(b, (int, float)):
                    self.baselines.append(BASELINES["constant"](b))
                else:
                    self.baselines.append(b)
        else:
            if isinstance(baselines, str):
                baseline = BASELINES[baselines]()
            elif isinstance(baselines, (int, float)):
                baseline = BASELINES["constant"](baselines)
            else:
                baseline = baselines
            self.baselines = ModuleList([deepcopy(baseline) for _ in range(num_heads)])

        self.num_negatives = num_negatives
        self.embedding_decay = embedding_decay
        self.symmetric_negatives = symmetric_negatives
        self.num_heads = num_heads

    def local_forward(
        self,
        z_x: Tuple[torch.Tensor],
        z_y: Tuple[torch.Tensor],
        y: Optional[torch.Tensor],
        idx: int,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        if self.kernel_scales[idx] == "auto":
            embedding_features = z_x[idx].shape[1]
            kernel_scale = np.sqrt(embedding_features)
        else:
            kernel_scale = self.kernel_scales[idx]

        logits = (
            self.kernels[idx](z_y[idx], z_x[idx], kernel_scale)
            * self.inverse_temperatures[idx]
        )
        local_pos_logits = diagonal(logits).unsqueeze(1)
        local_neg_logits = remove_diagonal(logits)
        if self.symmetric_negatives:
            logits = (
                self.kernels[idx](z_y[idx], z_y[idx], kernel_scale)
                * self.inverse_temperatures[idx]
            )
            local_neg_logits = torch.cat(
                [local_neg_logits, remove_diagonal(logits)], dim=-1
            )
        if self.num_negatives:
            local_neg_logits = random_sample_columns(
                local_neg_logits, self.num_negatives
            )
        local_log_baseline = self.baselines[idx](
            logits=local_neg_logits, y=y, z_y=z_y[idx]
        )
        local_pos_logits = local_pos_logits - local_log_baseline
        local_neg_logits = local_neg_logits - local_log_baseline
        return local_pos_logits, local_neg_logits, local_log_baseline

    def forward(
        self,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

        z_x = torch.chunk(z_x, self.num_heads, dim=-1)
        z_y = torch.chunk(z_y, self.num_heads, dim=-1)

        global_pos_logits = 0
        global_neg_logits = 0
        global_log_baseline = 0
        global_attraction = 0
        global_repulsion = 0

        for idx in range(self.num_heads):
            local_pos_logits, local_neg_logits, local_log_baseline = self.local_forward(
                z_x=z_x, z_y=z_y, y=y, idx=idx
            )
            local_attraction, local_repulsion = self.divergences[idx](
                local_pos_logits, local_neg_logits
            )

            global_pos_logits = global_pos_logits + local_pos_logits
            global_neg_logits = global_neg_logits + local_neg_logits
            global_log_baseline = global_log_baseline + local_log_baseline
            global_attraction = global_attraction + local_attraction
            global_repulsion = global_repulsion + local_repulsion

        global_pos_logits = global_pos_logits / self.num_heads
        global_neg_logits = global_neg_logits / self.num_heads
        global_log_baseline = global_log_baseline / self.num_heads
        global_attraction = global_attraction / self.num_heads
        global_repulsion = global_repulsion / self.num_heads

        with torch.no_grad():
            accuracy, recall, precision = classifier_metrics(
                global_pos_logits.flatten(), global_neg_logits.flatten()
            )
        return (
            global_pos_logits.mean(),
            global_neg_logits.mean(),
            global_pos_logits.sigmoid().mean(),
            global_neg_logits.sigmoid().mean(),
            accuracy,
            recall,
            precision,
            global_log_baseline.mean(),
            global_attraction.mean() + global_repulsion.mean() + embedding_decay,
        )


class ProductOfExpertsDensityRatioEstimator(MultiHeadDensityRatioEstimator):
    def __init__(
        self,
        num_heads: int,
        kernels: Union[str, Callable, List[Union[str, Callable]]] = "cauchy",
        kernel_scales: Union[float, str, List[Union[float, str]]] = 1.0,
        temperatures: Union[float, List[float]] = 1,
        divergence: Union[str, Callable] = "kld",
        baselines: Union[
            str, float, Callable, List[Union[str, float, Callable]]
        ] = "batch",
        num_negatives: Optional[int] = None,
        embedding_decay: float = 0,
        symmetric_negatives: bool = False,
    ) -> None:

        super().__init__(
            num_heads=num_heads,
            kernels=kernels,
            kernel_scales=kernel_scales,
            temperatures=temperatures,
            baselines=baselines,
            num_negatives=num_negatives,
            embedding_decay=embedding_decay,
            symmetric_negatives=symmetric_negatives,
        )

        self.divergence = (
            DIVERGENCES[divergence] if isinstance(divergence, str) else divergence
        )

    def forward(
        self,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

        z_x = torch.chunk(z_x, self.num_heads, dim=-1)
        z_y = torch.chunk(z_y, self.num_heads, dim=-1)

        global_pos_logits = 0
        global_neg_logits = 0
        global_log_baseline = 0

        for idx in range(self.num_heads):
            local_pos_logits, local_neg_logits, local_log_baseline = self.local_forward(
                z_x, z_y, y, idx
            )
            global_pos_logits = global_pos_logits + local_pos_logits
            global_neg_logits = global_neg_logits + local_neg_logits
            global_log_baseline = global_log_baseline + local_log_baseline

        global_pos_logits = global_pos_logits  # / self.num_heads
        global_neg_logits = global_neg_logits  # / self.num_heads
        global_log_baseline = global_log_baseline  # / self.num_heads
        global_attraction, global_repulsion = self.divergence(
            global_pos_logits, global_neg_logits
        )

        with torch.no_grad():
            accuracy, recall, precision = classifier_metrics(
                global_pos_logits.flatten(), global_neg_logits.flatten()
            )
        return (
            global_pos_logits.mean(),
            global_neg_logits.mean(),
            global_pos_logits.sigmoid().mean(),
            global_neg_logits.sigmoid().mean(),
            accuracy,
            recall,
            precision,
            global_log_baseline.mean(),
            global_attraction.mean() + global_repulsion.mean() + embedding_decay,
        )


class MixtureOfExpertsDensityRatioEstimator(ProductOfExpertsDensityRatioEstimator):
    def forward(
        self,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

        z_x = torch.chunk(z_x, self.num_heads, dim=-1)
        z_y = torch.chunk(z_y, self.num_heads, dim=-1)

        global_pos_logits = torch.zeros(1, device=z_x[0].device) - float("inf")
        global_neg_logits = torch.zeros(1, device=z_x[0].device) - float("inf")
        global_log_baseline = torch.zeros(1, device=z_x[0].device) - float("inf")

        for idx in range(self.num_heads):
            local_pos_logits, local_neg_logits, local_log_baseline = self.local_forward(
                z_x=z_x, z_y=z_y, y=y, idx=idx
            )
            global_pos_logits = torch.logaddexp(global_pos_logits, local_pos_logits)
            global_neg_logits = torch.logaddexp(global_neg_logits, local_neg_logits)
            global_log_baseline = torch.logaddexp(
                global_log_baseline, local_log_baseline
            )

        global_pos_logits = global_pos_logits - np.log(self.num_heads)
        global_neg_logits = global_neg_logits - np.log(self.num_heads)
        global_log_baseline = global_log_baseline - np.log(self.num_heads)
        global_attraction, global_repulsion = self.divergence(
            global_pos_logits, global_neg_logits
        )

        with torch.no_grad():
            accuracy, recall, precision = classifier_metrics(
                global_pos_logits.flatten(), global_neg_logits.flatten()
            )
        return (
            global_pos_logits.mean(),
            global_neg_logits.mean(),
            global_pos_logits.sigmoid().mean(),
            global_neg_logits.sigmoid().mean(),
            accuracy,
            recall,
            precision,
            global_log_baseline.mean(),
            global_attraction.mean() + global_repulsion.mean() + embedding_decay,
        )


class EncoderProjectorLoss(nn.Module):
    def __init__(self, encoder_loss, projector_loss):
        super().__init__()
        self.encoder_loss = encoder_loss
        self.projector_loss = projector_loss

    def forward(self, h_x, h_y, z_x, z_y, y, **kwargs):
        encoder_loss = self.encoder_loss(h_x, h_y, y)
        projector_loss = self.projector_loss(z_x, z_y, y)
        return tuple([(h + z) / 2 for (h, z) in zip(encoder_loss, projector_loss)])


class SymmetricLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, z_x, z_y, y, x, **kwargs):
        xy_loss = self.loss(z_x, z_y, y)
        yx_loss = self.loss(z_y, z_x, x)
        return tuple([(xy + yx) / 2 for (xy, yx) in zip(xy_loss, yx_loss)])


class SymmetricEncoderProjectorLoss(nn.Module):
    def __init__(self, encoder_loss, projector_loss):
        super().__init__()
        self.encoder_loss = encoder_loss
        self.projector_loss = projector_loss

    def forward(self, h_x, h_y, z_x, z_y, x, y, **kwargs):
        xy_encoder_loss = self.encoder_loss(h_x, h_y, y)
        xy_projector_loss = self.projector_loss(z_x, z_y, y)
        yx_encoder_loss = self.encoder_loss(h_y, h_x, x)
        yx_projector_loss = self.projector_loss(z_y, z_x, x)
        return tuple(
            [
                (xy_h + xy_z + yx_h + yx_z) / 4
                for (xy_h, xy_z, yx_h, yx_z) in zip(
                    xy_encoder_loss,
                    xy_projector_loss,
                    yx_encoder_loss,
                    yx_projector_loss,
                )
            ]
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
