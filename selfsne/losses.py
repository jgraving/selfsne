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

from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_precision,
    binary_recall,
)

from selfsne.kernels import PAIRWISE_KERNELS
from selfsne.divergences import DIVERGENCES
from selfsne.baselines import BASELINES
from selfsne.utils import (
    remove_diagonal,
    off_diagonal,
    stop_gradient,
    random_sample_columns,
)

from typing import Optional, Union, Tuple, List, Callable, Dict
from copy import deepcopy


# def classifier_metrics(pos_logits, neg_logits):
#     """
#     Calculates accuracy, recall, and precision given positive and negative logits.

#     Args:
#         pos_logits (torch.Tensor): A tensor of positive logits of shape (batch_size,).
#         neg_logits (torch.Tensor): A tensor of negative logits of shape (batch_size,).
#         threshold (float): A threshold to convert logits to binary predictions. Defaults to 0.5.

#     Returns:
#         accuracy (float): The accuracy score
#         recall (float): The recall score.
#         precision (float): The precision score.
#     """
#     # Combine positive and negative logits
#     logits = torch.cat((pos_logits, neg_logits), dim=0)

#     # Create binary labels (1 for positive, 0 for negative)
#     target = torch.cat(
#         (
#             torch.ones_like(pos_logits, dtype=torch.long, device=logits.device),
#             torch.zeros_like(neg_logits, dtype=torch.long, device=logits.device),
#         ),
#         dim=0,
#     )
#     return (
#         binary_accuracy(logits, target),
#         binary_recall(logits, target),
#         binary_precision(logits, target),
#     )


def classifier_metrics(pos_logits, neg_logits, threshold=True):
    """
    Calculates accuracy, recall/sensitivity, precision/PPV, NPV, and specificity given positive and negative logits.

    Args:
        pos_logits (torch.Tensor): A tensor of logits for the positive class of shape (batch_size,).
        neg_logits (torch.Tensor): A tensor of logits for the negative class of shape (batch_size,).
        threshold (bool): If True, apply hard thresholding to logits. If False, use the mean of sigmoid-transformed logits.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    dtype = pos_logits.dtype

    if threshold:
        # Apply hard thresholding to logits
        TP = (pos_logits > 0).to(dtype).mean()  # True Positive
        FP = (neg_logits > 0).to(dtype).mean()  # False Positive
        TN = (neg_logits <= 0).to(dtype).mean()  # True Negative
        FN = (pos_logits <= 0).to(dtype).mean()  # False Negative
    else:
        # Use mean of sigmoid-transformed logits
        pos_probs = torch.sigmoid(pos_logits).mean()
        neg_probs = torch.sigmoid(neg_logits).mean()
        TP = pos_probs  # True Positive probabilities for positive class
        FP = neg_probs  # False Positive probabilities for negative class
        TN = 1 - neg_probs  # True Negative probabilities for negative class
        FN = 1 - pos_probs  # False Negative probabilities for positive class

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    npv = TN / (TN + FN)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "npv": npv,
    }


class LikelihoodRatioEstimator(nn.Module):
    """
    A generalized contrastive loss for unnormalized likelihood ratio estimation based on
    Noise Contrastive Estimation (NCE). For representation learning the loss preserves embedding
    structure by maximizing similarity for positive pairs and minimizing similarity
    for negative (noise) pairs.

    A user-selected similarity kernel is used to calculate the log likelihood ratio ("logits")
    for each pair, which are then optimized by a user-selected divergence function to
    maximize logits for positive pairs (attraction) and minimize logits for negative pairs
    (repulsion).

    Args:
        kernel (Union[str, callable]): Similarity kernel used for calculating logits.
            Must be one of selfsne.kernels.KERNELS or a callable that takes in two 2D tensors and returns a pairwise 2D tensor.
            For example, "cauchy" can be used to produce t-SNE or UMAP embeddings, "normal" can be used to
            produce SNE embeddings, and "vonmises" can be used for (hyper)spherical embeddings.
        kernel_scale (Union[float, str]): Positive scale value for calculating logits.
            Can also be "parametric" to learn the scale as part of the model. Default is 1.0.
        temperature (Union[float, str]): The temperature for the logits. Larger values create more
            uniform embeddings. Can also be "parametric" to learn the temperature as part of the model.
            Default is 1.0.
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
        temperature: float = 1.0,
        divergence: Union[str, callable] = "kld",
        baseline: Union[str, float, callable] = "batch",
        num_negatives: Optional[int] = None,
        embedding_decay: float = 0,
        symmetric_negatives: bool = False,
        remove_neg_diagonal: bool = True,
        concat_chunk_encode: bool = False,
    ) -> None:

        super().__init__()
        if isinstance(kernel, str):
            self.kernel = PAIRWISE_KERNELS[kernel]
        else:
            self.kernel = kernel

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

        if isinstance(temperature, str):
            if temperature == "parametric":
                self.temperature = BASELINES["parametric"](activation=nn.Softplus())
            else:
                raise ValueError(
                    "Invalid string value for temperature. Only 'parametric' is allowed."
                )
        elif isinstance(temperature, (int, float)):
            self.temperature = BASELINES["constant"](1 / float(temperature))
        else:
            self.temperature = temperature  # You might consider adding validation or handling for other types

        # Check and set kernel_scale based on type
        if isinstance(kernel_scale, str):
            if kernel_scale == "parametric":
                self.kernel_scale = BASELINES["parametric"](activation=nn.Softplus())
            else:
                raise ValueError(
                    "Invalid string value for kernel_scale. Only 'parametric' is allowed."
                )
        elif isinstance(kernel_scale, (int, float)):
            self.kernel_scale = BASELINES["constant"](float(kernel_scale))
        else:
            self.kernel_scale = kernel_scale  # Similar to temperature, consider validation or handling for other types

        self.num_negatives = num_negatives
        self.embedding_decay = embedding_decay
        self.symmetric_negatives = symmetric_negatives
        self.remove_neg_diagonal = remove_neg_diagonal
        self.concat_chunk_encode = concat_chunk_encode

    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        encoder: nn.Module,
        projector: nn.Module,
        encoder_x: Optional[nn.Module] = None,
        projector_x: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if encoder_x is not None:
            h_x = encoder_x(x)
            z_x = projector_x(h_x)
            h_y = encoder(y)
            z_y = projector(h_y)
        else:
            if self.concat_chunk_encode:
                h_x, h_y = torch.chunk(encoder(torch.cat([x, y])), 2)
                z_x, z_y = torch.chunk(projector(torch.cat([h_x, h_y])), 2)
            else:
                h_x = encoder(x)
                h_y = encoder(y)
                z_x = projector(h_x)
                z_y = projector(h_y)

        return z_x, z_y

    def compute_logits(
        self,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        kernel_scale: torch.Tensor,
        inverse_temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logits = self.kernel(z_x, z_y, kernel_scale) * inverse_temperature
        pos_logits = diagonal(logits).unsqueeze(1)
        neg_logits = remove_diagonal(logits) if self.remove_neg_diagonal else logits

        if self.symmetric_negatives:
            logits = self.kernel(z_y, z_y, kernel_scale) * inverse_temperature
            neg_logits = torch.cat(
                [
                    neg_logits,
                    remove_diagonal(logits) if self.remove_neg_diagonal else logits,
                ],
                dim=-1,
            )

        if self.num_negatives:
            neg_logits = random_sample_columns(neg_logits, self.num_negatives)

        return pos_logits, neg_logits

    def compute_loss_and_metrics(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        z_y: torch.Tensor,
        log_baseline: torch.Tensor,
        kernel_scale: torch.Tensor,
        inverse_temperature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pos_logits = pos_logits - log_baseline
        neg_logits = neg_logits - log_baseline
        attraction, repulsion = self.divergence(pos_logits, neg_logits)

        embedding_prior = diagonal(
            self.kernel(z_y, torch.zeros_like(z_y), kernel_scale)
        ).unsqueeze(1)
        embedding_decay = -self.embedding_decay * embedding_prior.mean()

        with torch.no_grad():
            kld_attraction, kld_repulsion = DIVERGENCES["kld"](pos_logits, neg_logits)
            kld_lower_bound = kld_attraction + kld_repulsion
            rkld_attraction, rkld_repulsion = DIVERGENCES["rkld"](
                pos_logits, neg_logits
            )
            rkld_lower_bound = rkld_attraction + rkld_repulsion
            jsd_attraction, jsd_repulsion = DIVERGENCES["jsd"](pos_logits, neg_logits)
            jsd_lower_bound = jsd_attraction + jsd_repulsion
            metrics = classifier_metrics(pos_logits.flatten(), neg_logits.flatten())

        lower_bound = attraction.mean() + repulsion.mean()
        loss = lower_bound + embedding_decay

        return {
            "pos_logits": pos_logits.mean(),
            "neg_logits": neg_logits.mean(),
            "pos_prob": pos_logits.sigmoid().mean(),
            "neg_prob": neg_logits.sigmoid().mean(),
            "log_baseline": log_baseline.mean(),
            "inverse_temperature": inverse_temperature.mean(),
            "kernel_scale": kernel_scale.mean(),
            "lower_bound": -lower_bound,
            "kld": -kld_lower_bound,
            "rkld": -rkld_lower_bound,
            "jsd": -jsd_lower_bound,
            "embedding_decay": embedding_decay,
            "loss": loss,
            **metrics,
        }

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        encoder: nn.Module,
        projector: nn.Module,
        encoder_x: Optional[nn.Module] = None,
        projector_x: Optional[nn.Module] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        z_x, z_y = self.encode(
            x=x,
            y=y,
            encoder=encoder,
            projector=projector,
            encoder_x=encoder_x,
            projector_x=projector_x,
        )

        kernel_scale = self.kernel_scale(z_y=z_y)
        inverse_temperature = self.temperature(z_y=z_y)

        pos_logits, neg_logits = self.compute_logits(
            z_x=z_x,
            z_y=z_y,
            kernel_scale=kernel_scale,
            inverse_temperature=inverse_temperature,
        )

        log_baseline = self.baseline(
            pos_logits=pos_logits, neg_logits=neg_logits, y=y, z_y=z_y
        )

        return self.compute_loss_and_metrics(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            z_y=z_y,
            log_baseline=log_baseline,
            kernel_scale=kernel_scale,
            inverse_temperature=inverse_temperature,
        )


DensityRatioEstimator = LikelihoodRatioEstimator


class EncoderProjectorLoss(nn.Module):
    def __init__(self, encoder_loss, projector_loss):
        super().__init__()
        self.encoder_loss = encoder_loss
        self.projector_loss = projector_loss
        self.identity = nn.Identity()

    def forward(
        self, x, y, encoder, projector, encoder_x=None, projector_x=None, **kwargs
    ):
        # Encode once to get h_x and h_y
        h_x, h_y = self.encoder_loss.encode(
            x=x,
            y=y,
            encoder=encoder,
            projector=self.identity,
            encoder_x=encoder_x,
            projector_x=self.identity,
        )

        # Apply the projector to h_x and h_y to get z_x and z_y
        z_x, z_y = self.encoder_loss.encode(
            x=h_x,
            y=h_y,
            encoder=self.identity,
            projector=projector,
            encoder_x=self.identity,
            projector_x=projector_x,
        )

        encoder_loss_result = self.encoder_loss(
            x=h_x,
            y=h_y,
            encoder=self.identity,
            projector=self.identity,
            **kwargs,
        )
        projector_loss_result = self.projector_loss(
            x=z_x,
            y=z_y,
            encoder=self.identity,
            projector=self.identity,
            **kwargs,
        )
        combined_loss = {}
        for key in encoder_loss_result.keys():
            combined_loss[key] = (
                encoder_loss_result[key] + projector_loss_result[key]
            ) / 2
        return combined_loss


class SymmetricLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.identity = nn.Identity()

    def forward(
        self, x, y, encoder, projector, encoder_x=None, projector_x=None, **kwargs
    ):
        # Encode once to get z_x and z_y
        z_x, z_y = self.loss.encode(
            x=x,
            y=y,
            encoder=encoder,
            projector=projector,
            encoder_x=encoder_x,
            projector_x=projector_x,
        )

        xy_loss = self.loss(
            x=z_x,
            y=z_y,
            encoder=self.identity,
            projector=self.identity,
            **kwargs,
        )
        yx_loss = self.loss(
            x=z_y,
            y=z_x,
            encoder=self.identity,
            projector=self.identity,
            **kwargs,
        )
        symmetric_loss = {}
        for key in xy_loss.keys():
            symmetric_loss[key] = (xy_loss[key] + yx_loss[key]) / 2
        return symmetric_loss


class RedundancyReduction(nn.Module):
    """
    Redundancy Reduction loss function that creates an embedding by minimizing
    mean squared error between an identity matrix and the empirical cross-correlation matrix
    between positive pairs. This helps to maximize feature invariance to differences between
    positive pairs while minimizing redundancy (correlation) between features, preserving global
    structure in the embedding as a form of nonlinear Canonical Correlation Analysis (CCA).

    Args:
        num_features (int): Number of embedding features.
        invariance_weight (float, optional): Weighting for the invariance term in the loss.
            Default: 1.0.
        redundancy_weight (float, optional): Weighting for the redundancy term in the loss.
            Default: 1.0.

    Returns:
        loss_dict (dict): A dictionary containing the computed redundancy reduction loss components.
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _ = x.shape

        correlation = self.norm(x).T @ self.norm(y) / batch_size
        invariance = diagonal(correlation).add_(-1).pow_(2).mean()
        redundancy = off_diagonal(correlation).pow_(2).mean()

        loss = invariance * self.invariance_weight + redundancy * self.redundancy_weight

        return {
            "loss": loss,
            "invariance": invariance,
            "redundancy": redundancy,
        }


class VICReg(nn.Module):
    """
    Variance-Invariance-Covariance (VIC) regularization loss function that creates an
    embedding by combining three terms:
    (1) a variance stabilizing term (hinge loss for feature-wise std. dev.),
    (2) an invariance term (mean squared error between positive pairs),
    (3) a covariance regularization term (minimize squared covariance) based on redundancy reduction.
    This helps to preserve global structure in the embedding as a form of Laplacian Eigenmaps.

    Args:
        num_features (int): Number of embedding features.
        eps (float, optional): A value added to the variance term for numerical stability. Default: 1e-8.
        variance_weight (float, optional): Weighting for the variance term in the loss. Default: 1.0.
        invariance_weight (float, optional): Weighting for the invariance term in the loss. Default: 1.0.
        covariance_weight (float, optional): Weighting for the covariance term in the loss. Default: 1.0.

    Returns:
        loss_dict (dict): A dictionary containing the computed VICReg loss components.
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        loss = (
            variance * self.variance_weight
            + invariance * self.invariance_weight
            + covariance * self.covariance_weight
        )

        return {
            "loss": loss,
            "variance": variance,
            "invariance": invariance,
            "covariance": covariance,
        }


# class TRELLIS(nn.Module):
#     def __init__(
#         self,
#         num_waymarks: int = 1,
#         kernel: Union[str, callable] = "cauchy",
#         kernel_scale: Union[float, str] = 1.0,
#         temperature: float = 1,
#         divergence: Union[str, callable] = "kld",
#         baseline: Union[str, float, callable] = "batch",
#         num_negatives: Optional[int] = None,
#         embedding_decay: float = 0,
#         symmetric_negatives: bool = False,
#         remove_neg_diagonal: bool = True,
#     ) -> None:

#         super().__init__()
#         self.num_waymarks = num_waymarks
#         self.num_estimators = num_waymarks + 1
#         if isinstance(kernel, str):
#             self.kernel = PAIRWISE_KERNELS[kernel]
#         else:
#             self.kernel = kernel

#         self.kernel_scale = kernel_scale

#         if isinstance(divergence, str):
#             self.divergence = DIVERGENCES[divergence]
#         else:
#             self.divergence = divergence

#         if isinstance(baseline, str):
#             baseline = BASELINES[baseline]()
#         elif isinstance(baseline, (int, float)):
#             baseline = BASELINES["constant"](baseline)

#         self.baselines = ModuleList(
#             [deepcopy(baseline) for _ in range(self.num_estimators)]
#         )
#         self.num_negatives = num_negatives
#         self.symmetric_negatives = symmetric_negatives
#         self.embedding_decay = embedding_decay
#         self.inverse_temperature = 1 / temperature
#         if self.num_estimators > 1:
#             self.register_buffer(
#                 "alphas", torch.linspace(0, 1, self.num_waymarks + 2)[1:-1]
#             )
#         self.remove_neg_diagonal = remove_neg_diagonal

#     def forward(
#         self,
#         z_x: torch.Tensor,
#         z_y: torch.Tensor,
#         y: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor]:
#         embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

#         if self.kernel_scale == "auto":
#             embedding_features = z_x.shape[1]
#             self.kernel_scale = np.sqrt(embedding_features)

#         logits = self.kernel(z_y, z_x, self.kernel_scale) * self.inverse_temperature
#         pos_logits = diagonal(logits).unsqueeze(1)
#         neg_logits = remove_diagonal(logits) if self.remove_neg_diagonal else logits
#         if self.symmetric_negatives:
#             logits = self.kernel(z_y, z_y, self.kernel_scale) * self.inverse_temperature
#             neg_logits = torch.cat(
#                 [
#                     neg_logits,
#                     remove_diagonal(logits) if self.remove_neg_diagonal else logits,
#                 ],
#                 dim=-1,
#             )
#         if self.num_negatives:
#             neg_logits = random_sample_columns(neg_logits, self.num_negatives)
#         if self.num_estimators > 1:
#             waymark_logits = log_lerp(
#                 pos_logits, neg_logits, self.alphas.view(-1, 1, 1)
#             )
#             waymark_logits = torch.cat(
#                 [
#                     pos_logits.expand_as(waymark_logits[0, None]),
#                     waymark_logits,
#                     neg_logits.expand_as(waymark_logits[0, None]),
#                 ],
#                 dim=0,
#             )
#             log_baselines = [
#                 self.baselines[idx](
#                     logits=waymark_logits[idx + 1], y=y, z_y=z_y.squeeze(1)
#                 )
#                 for idx in range(self.num_estimators)
#             ]
#         else:
#             waymark_logits = torch.stack(
#                 [
#                     pos_logits.expand_as(neg_logits),
#                     neg_logits,
#                 ],
#                 dim=0,
#             )
#             log_baselines = [
#                 self.baselines[idx](
#                     logits=waymark_logits[idx + 1], y=y, z_y=z_y.squeeze(1)
#                 )
#                 for idx in range(self.num_estimators)
#             ]

#         global_pos_logits = 0
#         global_neg_logits = 0
#         global_attraction = 0
#         global_repulsion = 0
#         global_log_baseline = 0

#         for idx in range(self.num_estimators):
#             local_log_baseline = log_baselines[idx]
#             local_pos_logits = waymark_logits[idx] - local_log_baseline
#             local_neg_logits = waymark_logits[idx + 1] - local_log_baseline
#             local_attraction, local_repulsion = self.divergence(
#                 local_pos_logits, local_neg_logits
#             )
#             global_pos_logits = global_pos_logits + local_pos_logits
#             global_neg_logits = global_neg_logits + local_neg_logits
#             global_attraction = global_attraction + local_attraction
#             global_repulsion = global_repulsion + local_repulsion
#             global_log_baseline = global_log_baseline + local_log_baseline

#         with torch.no_grad():
#             accuracy, recall, precision = classifier_metrics(
#                 logmeanexp(global_pos_logits, -1).flatten(), global_neg_logits.flatten()
#             )
#         return (
#             global_pos_logits.mean(),
#             global_neg_logits.mean(),
#             global_pos_logits.sigmoid().mean(),
#             global_neg_logits.sigmoid().mean(),
#             accuracy,
#             recall,
#             precision,
#             global_log_baseline.mean(),
#             global_attraction.mean() + global_repulsion.mean() + embedding_decay,
#         )

# class MultiheadDensityRatioEstimator(nn.Module):
#     def __init__(
#         self,
#         num_heads: int,
#         kernels: Union[str, Callable, List[Union[str, Callable]]] = "cauchy",
#         kernel_scale: Union[float, str, List[Union[float, str]]] = 1.0,
#         temperature: Union[float, List[float]] = 1,
#         divergence: Union[str, Callable, List[Union[str, Callable]]] = "kld",
#         baseline: Union[
#             str, float, Callable, List[Union[str, float, Callable]]
#         ] = "batch",
#         num_negatives: Optional[int] = None,
#         embedding_decay: float = 0,
#         symmetric_negatives: bool = False,
#     ) -> None:

#         super().__init__()

#         if isinstance(kernel, list):
#             assert (
#                 len(kernel) == num_heads
#             ), "Length of kernels list should match num_heads"
#             self.kernel = [
#                 PAIRWISE_KERNELS[k] if isinstance(k, str) else k for k in kernel
#             ]
#         else:
#             kernel = PAIRWISE_KERNELS[kernel] if isinstance(kernel, str) else kernel
#             self.kernel = [kernel] * num_heads

#         if isinstance(kernel_scale, list):
#             assert (
#                 len(kernel_scale) == num_heads
#             ), "Length of kernel_scale list should match num_heads"
#             self.kernel_scale = kernel_scale
#         else:
#             self.kernel_scale = [kernel_scale] * num_heads

#         if isinstance(temperature, list):
#             assert (
#                 len(temperature) == num_heads
#             ), "Length of temperature list should match num_heads"
#             self.inverse_temperature = [1 / t for t in temperature]
#         else:
#             self.inverse_temperature = [1 / temperature] * num_heads

#         if isinstance(divergence, list):
#             assert (
#                 len(divergence) == num_heads
#             ), "Length of divergence list should match num_heads"
#             self.divergence = [
#                 DIVERGENCES[d] if isinstance(d, str) else d for d in divergence
#             ]
#         else:
#             divergence = (
#                 DIVERGENCES[divergence] if isinstance(divergence, str) else divergence
#             )
#             self.divergence = [divergence] * num_heads

#         if isinstance(baseline, list):
#             assert (
#                 len(baseline) == num_heads
#             ), "Length of baseline list should match num_heads"
#             self.baseline = ModuleList()
#             for b in baseline:
#                 if isinstance(b, str):
#                     self.baseline.append(BASELINES[b]())
#                 elif isinstance(b, (int, float)):
#                     self.baseline.append(BASELINES["constant"](b))
#                 else:
#                     self.baseline.append(b)
#         else:
#             if isinstance(baseline, str):
#                 baseline = BASELINES[baseline]()
#             elif isinstance(baseline, (int, float)):
#                 baseline = BASELINES["constant"](baseline)
#             self.baseline = ModuleList([deepcopy(baseline) for _ in range(num_heads)])

#         self.num_negatives = num_negatives
#         self.embedding_decay = embedding_decay
#         self.symmetric_negatives = symmetric_negatives
#         self.num_heads = num_heads

#     def local_forward(
#         self,
#         z_x: Tuple[torch.Tensor],
#         z_y: Tuple[torch.Tensor],
#         y: Optional[torch.Tensor],
#         idx: int,
#         **kwargs,
#     ) -> Tuple[torch.Tensor]:
#         if self.kernel_scale[idx] == "auto":
#             embedding_features = z_x[idx].shape[1]
#             kernel_scale = np.sqrt(embedding_features)
#         else:
#             kernel_scale = self.kernel_scale[idx]

#         logits = (
#             self.kernel[idx](z_y[:, idx], z_x[:, idx], kernel_scale)
#             * self.inverse_temperature[idx]
#         )
#         local_pos_logits = diagonal(logits).unsqueeze(1)
#         local_neg_logits = remove_diagonal(logits)
#         if self.symmetric_negatives:
#             logits = (
#                 self.kernel[idx](z_y[:, idx], z_y[:, idx], kernel_scale)
#                 * self.inverse_temperature[idx]
#             )
#             local_neg_logits = torch.cat(
#                 [local_neg_logits, remove_diagonal(logits)], dim=-1
#             )
#         if self.num_negatives:
#             local_neg_logits = random_sample_columns(
#                 local_neg_logits, self.num_negatives
#             )
#         local_log_baseline = self.baseline[idx](
#             logits=local_neg_logits, y=y, z_y=z_y[:, idx]
#         )
#         local_pos_logits = local_pos_logits - local_log_baseline
#         local_neg_logits = local_neg_logits - local_log_baseline
#         return local_pos_logits, local_neg_logits, local_log_baseline

#     def forward(
#         self,
#         z_x: torch.Tensor,
#         z_y: torch.Tensor,
#         y: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor]:
#         if z_x.dim() != 3 or z_y.dim() != 3:
#             raise ValueError("z_x and z_y must be 3D tensors for multihead loss")
#         if z_x.shape[1] != self.num_heads or z_y.shape[1] != self.num_heads:
#             raise ValueError("z_x and z_y must have shape[1] equal to self.num_heads")

#         embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

#         global_pos_logits = 0
#         global_neg_logits = 0
#         global_log_baseline = 0
#         global_attraction = 0
#         global_repulsion = 0

#         for idx in range(self.num_heads):
#             local_pos_logits, local_neg_logits, local_log_baseline = self.local_forward(
#                 z_x=z_x, z_y=z_y, y=y, idx=idx
#             )
#             local_attraction, local_repulsion = self.divergence[idx](
#                 local_pos_logits, local_neg_logits
#             )

#             global_pos_logits = global_pos_logits + local_pos_logits
#             global_neg_logits = global_neg_logits + local_neg_logits
#             global_log_baseline = global_log_baseline + local_log_baseline
#             global_attraction = global_attraction + local_attraction
#             global_repulsion = global_repulsion + local_repulsion

#         global_pos_logits = global_pos_logits / self.num_heads
#         global_neg_logits = global_neg_logits / self.num_heads
#         global_log_baseline = global_log_baseline / self.num_heads
#         global_attraction = global_attraction / self.num_heads
#         global_repulsion = global_repulsion / self.num_heads

#         with torch.no_grad():
#             accuracy, recall, precision = classifier_metrics(
#                 global_pos_logits.flatten(), global_neg_logits.flatten()
#             )
#         return (
#             global_pos_logits.mean(),
#             global_neg_logits.mean(),
#             global_pos_logits.sigmoid().mean(),
#             global_neg_logits.sigmoid().mean(),
#             accuracy,
#             recall,
#             precision,
#             global_log_baseline.mean(),
#             global_attraction.mean() + global_repulsion.mean() + embedding_decay,
#         )


# class ProductOfExpertsDensityRatioEstimator(MultiheadDensityRatioEstimator):
#     def __init__(
#         self,
#         num_heads: int,
#         kernel: Union[str, Callable, List[Union[str, Callable]]] = "cauchy",
#         kernel_scale: Union[float, str, List[Union[float, str]]] = 1.0,
#         temperature: Union[float, List[float]] = 1,
#         divergence: Union[str, Callable] = "kld",
#         baseline: Union[
#             str, float, Callable, List[Union[str, float, Callable]]
#         ] = "batch",
#         num_negatives: Optional[int] = None,
#         embedding_decay: float = 0,
#         symmetric_negatives: bool = False,
#     ) -> None:

#         super().__init__(
#             num_heads=num_heads,
#             kernel=kernel,
#             kernel_scale=kernel_scale,
#             temperature=temperature,
#             baseline=baseline,
#             num_negatives=num_negatives,
#             embedding_decay=embedding_decay,
#             symmetric_negatives=symmetric_negatives,
#         )

#         self.divergence = (
#             DIVERGENCES[divergence] if isinstance(divergence, str) else divergence
#         )

#     def forward(
#         self,
#         z_x: torch.Tensor,
#         z_y: torch.Tensor,
#         y: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor]:

#         if z_x.dim() != 3 or z_y.dim() != 3:
#             raise ValueError("z_x and z_y must be 3D tensors for multihead loss")
#         if z_x.shape[1] != self.num_heads or z_y.shape[1] != self.num_heads:
#             raise ValueError("z_x and z_y must have shape[1] equal to self.num_heads")

#         embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

#         global_pos_logits = 0
#         global_neg_logits = 0
#         global_log_baseline = 0

#         for idx in range(self.num_heads):
#             local_pos_logits, local_neg_logits, local_log_baseline = self.local_forward(
#                 z_x, z_y, y, idx
#             )
#             global_pos_logits = global_pos_logits + local_pos_logits
#             global_neg_logits = global_neg_logits + local_neg_logits
#             global_log_baseline = global_log_baseline + local_log_baseline

#         global_pos_logits = global_pos_logits / self.num_heads
#         global_neg_logits = global_neg_logits / self.num_heads
#         global_log_baseline = global_log_baseline / self.num_heads
#         global_attraction, global_repulsion = self.divergence(
#             global_pos_logits, global_neg_logits
#         )

#         with torch.no_grad():
#             accuracy, recall, precision = classifier_metrics(
#                 global_pos_logits.flatten(), global_neg_logits.flatten()
#             )
#         return (
#             global_pos_logits.mean(),
#             global_neg_logits.mean(),
#             global_pos_logits.sigmoid().mean(),
#             global_neg_logits.sigmoid().mean(),
#             accuracy,
#             recall,
#             precision,
#             global_log_baseline.mean(),
#             global_attraction.mean() + global_repulsion.mean() + embedding_decay,
#         )


# class MixtureOfExpertsDensityRatioEstimator(ProductOfExpertsDensityRatioEstimator):
#     def forward(
#         self,
#         z_x: torch.Tensor,
#         z_y: torch.Tensor,
#         y: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor]:
#         if z_x.dim() != 3 or z_y.dim() != 3:
#             raise ValueError("z_x and z_y must be 3D tensors for multihead loss")
#         if z_x.shape[1] != self.num_heads or z_y.shape[1] != self.num_heads:
#             raise ValueError("z_x and z_y must have shape[1] equal to self.num_heads")

#         embedding_decay = self.embedding_decay * (z_x.pow(2).mean() + z_y.pow(2).mean())

#         global_pos_logits = torch.zeros(1, device=z_x[0].device) - float("inf")
#         global_neg_logits = torch.zeros(1, device=z_x[0].device) - float("inf")
#         global_log_baseline = torch.zeros(1, device=z_x[0].device) - float("inf")

#         for idx in range(self.num_heads):
#             local_pos_logits, local_neg_logits, local_log_baseline = self.local_forward(
#                 z_x=z_x, z_y=z_y, y=y, idx=idx
#             )
#             global_pos_logits = torch.logaddexp(global_pos_logits, local_pos_logits)
#             global_neg_logits = torch.logaddexp(global_neg_logits, local_neg_logits)
#             global_log_baseline = torch.logaddexp(
#                 global_log_baseline, local_log_baseline
#             )

#         global_pos_logits = global_pos_logits - np.log(self.num_heads)
#         global_neg_logits = global_neg_logits - np.log(self.num_heads)
#         global_log_baseline = global_log_baseline - np.log(self.num_heads)
#         global_attraction, global_repulsion = self.divergence(
#             global_pos_logits, global_neg_logits
#         )

#         with torch.no_grad():
#             accuracy, recall, precision = classifier_metrics(
#                 global_pos_logits.flatten(), global_neg_logits.flatten()
#             )
#         return (
#             global_pos_logits.mean(),
#             global_neg_logits.mean(),
#             global_pos_logits.sigmoid().mean(),
#             global_neg_logits.sigmoid().mean(),
#             accuracy,
#             recall,
#             precision,
#             global_log_baseline.mean(),
#             global_attraction.mean() + global_repulsion.mean() + embedding_decay,
#         )
