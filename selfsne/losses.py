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
from torch import nn, diagonal
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
    ALL_METRICS = [
        "loss",
        "pos_logits",
        "neg_logits",
        "pos_prob",
        "neg_prob",
        "pos_similarity",
        "neg_similarity",
        "attraction",
        "repulsion",
        "log_baseline",
    ]

    def __init__(
        self,
        kernel: Union[str, callable] = "cauchy",
        kernel_scale: Union[float, int] = 1.0,
        divergence: Union[str, callable] = "kld",
        baseline: Union[str, float, callable] = "parametric",
        classifier_metrics: bool = False,
        pos_as_neg: bool = False,
        metrics: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.kernel = ROWWISE_KERNELS[kernel] if isinstance(kernel, str) else kernel
        self.divergence = (
            DIVERGENCES[divergence] if isinstance(divergence, str) else divergence
        )

        if isinstance(baseline, str):
            self.baseline = BASELINES[baseline]()
        elif isinstance(baseline, (int, float)):
            self.baseline = BASELINES["constant"](baseline)
        else:
            self.baseline = baseline

        self.kernel_scale = float(kernel_scale)
        self.pos_as_neg = pos_as_neg
        self.classifier_metrics = classifier_metrics
        self.metrics = self._expand_metrics(metrics)

    def _expand_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        if metrics is None:
            return ["loss"]

        expanded = set()
        for m in metrics:
            m = m.lower()
            if m in {"logit", "logits"}:
                expanded.update(["pos_logits", "neg_logits"])
            elif m in {"prob", "probs", "probabilities"}:
                expanded.update(["pos_prob", "neg_prob"])
            elif m in {"similarity", "similarities"}:
                expanded.update(["pos_similarity", "neg_similarity"])
            elif m == "all":
                expanded.update(self.ALL_METRICS)
            else:
                expanded.add(m)

        return list(expanded)

    def similarities(
        self,
        context_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        kernel_scale: float,
        reference_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if reference_embedding is not None:
            pos_similarity = self.kernel(
                context_embedding, target_embedding, kernel_scale
            ).unsqueeze(1)
            neg_similarity = self.kernel(
                context_embedding.unsqueeze(1), reference_embedding, kernel_scale
            )
        else:
            similarity = self.kernel(
                context_embedding.unsqueeze(1), target_embedding, kernel_scale
            )
            pos_similarity = diagonal(similarity).unsqueeze(1)
            neg_similarity = remove_diagonal(similarity)
        if self.pos_as_neg:
            neg_similarity = torch.cat([neg_similarity, pos_similarity], dim=1)
        return pos_similarity, neg_similarity

    def loss_and_metrics(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        log_baseline: torch.Tensor,
        pos_similarity: torch.Tensor,
        neg_similarity: torch.Tensor,
        attraction: torch.Tensor,
        repulsion: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.training:
            return {"loss": attraction + repulsion}

        out = {"loss": attraction + repulsion}

        if "pos_logits" in self.metrics:
            out["pos_logits"] = pos_logits.mean()
        if "neg_logits" in self.metrics:
            out["neg_logits"] = neg_logits.mean()
        if "pos_prob" in self.metrics:
            out["pos_prob"] = pos_logits.sigmoid().mean()
        if "neg_prob" in self.metrics:
            out["neg_prob"] = neg_logits.sigmoid().mean()
        if "pos_similarity" in self.metrics:
            out["pos_similarity"] = pos_similarity.mean()
        if "neg_similarity" in self.metrics:
            out["neg_similarity"] = neg_similarity.mean()
        if "attraction" in self.metrics:
            out["attraction"] = attraction
        if "repulsion" in self.metrics:
            out["repulsion"] = repulsion
        if "log_baseline" in self.metrics:
            out["log_baseline"] = log_baseline.mean()

        if self.classifier_metrics:
            clf = classifier_metrics(pos_logits.flatten(), neg_logits.flatten())
            for k in self.metrics:
                if k in clf:
                    out[k] = clf[k]

        return out

    def forward(
        self,
        context_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        reference_embedding: Optional[torch.Tensor] = None,
        baseline_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.predict_logits(
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            reference_embedding=reference_embedding,
            baseline_embedding=baseline_embedding,
        )

        attraction, repulsion = self.divergence(
            output["pos_logits"], output["neg_logits"]
        )

        return self.loss_and_metrics(
            pos_logits=output["pos_logits"],
            neg_logits=output["neg_logits"],
            log_baseline=output["log_baseline"],
            pos_similarity=output["pos_similarity"],
            neg_similarity=output["neg_similarity"],
            attraction=attraction,
            repulsion=repulsion,
        )

    def predict_logits(
        self,
        context_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        reference_embedding: Optional[torch.Tensor] = None,
        baseline_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pos_similarity, neg_similarity = self.similarities(
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            kernel_scale=self.kernel_scale,
            reference_embedding=reference_embedding,
        )
        log_baseline = self.baseline(
            pos_similarity=pos_similarity,
            neg_similarity=neg_similarity,
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            baseline_embedding=baseline_embedding,
            kernel=self.kernel,
            kernel_scale=self.kernel_scale,
        )
        pos_logits = pos_similarity - log_baseline
        neg_logits = neg_similarity - log_baseline
        return {
            "pos_similarity": pos_similarity,
            "neg_similarity": neg_similarity,
            "log_baseline": log_baseline,
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
        }
