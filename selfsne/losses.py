# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch import nn
from torch import diagonal

from selfsne.kernels import ROWWISE_KERNELS
from selfsne.divergences import DIVERGENCES
from selfsne.baselines import BASELINES
from selfsne.utils import remove_diagonal

from typing import Optional, Union, Tuple, Dict


def classifier_metrics(pos_logits, neg_logits, threshold=True):
    dtype = pos_logits.dtype
    if threshold:
        TP = (pos_logits > 0).to(dtype).mean()
        FP = (neg_logits > 0).to(dtype).mean()
        TN = (neg_logits <= 0).to(dtype).mean()
        FN = (pos_logits <= 0).to(dtype).mean()
    else:
        pos_probs = torch.sigmoid(pos_logits).mean()
        neg_probs = torch.sigmoid(neg_logits).mean()
        TP = pos_probs
        FP = neg_probs
        TN = 1 - neg_probs
        FN = 1 - pos_probs
    accuracy = (TP + TN) / 2
    precision = TP / (TP + FP)
    npv = TN / (TN + FN)
    apv = (precision + npv) / 2
    recall = TP
    specificity = TN
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "npv": npv,
        "apv": apv,
    }


class LikelihoodRatioEstimator(nn.Module):
    def __init__(
        self,
        kernel: Union[str, callable] = "cauchy",
        kernel_scale: Union[float, int] = 1.0,
        divergence: Union[str, callable] = "kld",
        baseline: Union[str, float, callable] = "batch",
    ) -> None:
        super().__init__()
        if isinstance(kernel, str):
            self.kernel = ROWWISE_KERNELS[kernel]
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

        self.kernel_scale = float(kernel_scale)

    def logits(
        self,
        context_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        kernel_scale: float,
        reference_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if reference_embedding is not None:
            pos_logits = self.kernel(context_embedding, target_embedding, kernel_scale)
            neg_logits = self.kernel(
                context_embedding.unsqueeze(1), reference_embedding, kernel_scale
            )
        else:
            logits = self.kernel(
                context_embedding.unsqueeze(1), target_embedding, kernel_scale
            )
            pos_logits = diagonal(logits).unsqueeze(1)
            neg_logits = remove_diagonal(logits)

        return pos_logits, neg_logits

    def loss_and_metrics(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        log_baseline: torch.Tensor,
        kernel_scale: float,
    ) -> Dict[str, torch.Tensor]:
        pos_logits = pos_logits - log_baseline
        neg_logits = neg_logits - log_baseline
        attraction, repulsion = self.divergence(pos_logits, neg_logits)
        metrics = classifier_metrics(pos_logits.flatten(), neg_logits.flatten())
        loss = attraction.mean() + repulsion.mean()
        return {
            "pos_logits": pos_logits.mean(),
            "neg_logits": neg_logits.mean(),
            "pos_prob": pos_logits.sigmoid().mean(),
            "neg_prob": neg_logits.sigmoid().mean(),
            "attraction": attraction,
            "repulsion": repulsion,
            "log_baseline": log_baseline.mean(),
            "loss": loss,
            **metrics,
        }

    def forward(
        self,
        context_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        reference_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pos_logits, neg_logits = self.logits(
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            kernel_scale=self.kernel_scale,
            reference_embedding=reference_embedding,
        )
        log_baseline = self.baseline(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            context_embedding=context_embedding,
            target_embedding=target_embedding,
        )
        return self.loss_and_metrics(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            log_baseline=log_baseline,
            kernel_scale=self.kernel_scale,
        )
