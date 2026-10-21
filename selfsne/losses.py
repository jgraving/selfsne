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

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn, diagonal
from selfsne.kernels import ROWWISE_KERNELS
from selfsne.divergences import DIVERGENCES
from selfsne.baselines import BASELINES
from selfsne.utils import remove_diagonal
from typing import Optional, Union, Tuple, Dict, List


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
    # A classifier making no positive (or negative) predictions has undefined
    # precision (or NPV); report zero instead of poisoning all logged metrics.
    precision = TP / (TP + FP).clamp_min(torch.finfo(dtype).tiny)
    npv = TN / (TN + FN).clamp_min(torch.finfo(dtype).tiny)
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

        for name, value, choices in (
            ("kernel", kernel, ROWWISE_KERNELS),
            ("divergence", divergence, DIVERGENCES),
            ("baseline", baseline, BASELINES),
        ):
            if isinstance(value, str) and value not in choices:
                raise ValueError(
                    f"Unknown {name} {value!r}. Choose from: {', '.join(sorted(choices))}."
                )
        if not math.isfinite(float(kernel_scale)) or float(kernel_scale) <= 0:
            raise ValueError("kernel_scale must be finite and greater than zero.")
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

        self._builtin_baseline = isinstance(baseline, (str, int, float))
        self.kernel_scale = float(kernel_scale)
        self.pos_as_neg = pos_as_neg
        self.classifier_metrics = classifier_metrics
        self.metrics = self._expand_metrics(metrics)

    def _checkpoint_signature(self):
        """Describe supported built-in loss settings not contained in weights."""
        if (
            not self._builtin_baseline
            or not any(self.kernel is kernel for kernel in ROWWISE_KERNELS.values())
            or not any(
                self.divergence is divergence for divergence in DIVERGENCES.values()
            )
            or not hasattr(self.kernel, "__name__")
            or not hasattr(self.divergence, "__name__")
        ):
            return None
        return {
            "kernel": self.kernel.__name__,
            "kernel_scale": self.kernel_scale,
            "divergence": self.divergence.__name__,
            "baseline": type(self.baseline).__name__,
            "pos_as_neg": bool(self.pos_as_neg),
        }

    def _expand_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        if metrics is None:
            return ["loss"]
        if isinstance(metrics, str):
            metrics = [metrics]
        classifier_names = {
            "accuracy",
            "recall",
            "precision",
            "specificity",
            "npv",
            "apv",
        }
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
                if self.classifier_metrics:
                    expanded.update(classifier_names)
            elif m in self.ALL_METRICS or (
                self.classifier_metrics and m in classifier_names
            ):
                expanded.add(m)
            else:
                raise ValueError(f"Unknown or disabled metric {m!r}.")

        return sorted(expanded)

    def _kernel_call(self, x, y, scale):
        if not any(self.kernel is kernel for kernel in ROWWISE_KERNELS.values()):
            return self.kernel(x, y, scale)
        # Distance reductions can overflow in float16; angular/inverse kernels
        # also need float32 epsilon and reduction precision under autocast.
        context = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            if x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
            if y.dtype in (torch.float16, torch.bfloat16):
                y = y.float()
            return self.kernel(x, y, scale)

    def similarities(
        self,
        context_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        kernel_scale: float,
        reference_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context_embedding.ndim != 2 or target_embedding.ndim != 2:
            raise ValueError(
                "context and target embeddings must have shape (batch, features)."
            )
        if (
            context_embedding.shape != target_embedding.shape
            or context_embedding.numel() == 0
        ):
            raise ValueError(
                "context and target embeddings must have matching, non-empty shapes."
            )
        if (
            not context_embedding.is_floating_point()
            or not target_embedding.is_floating_point()
        ):
            raise TypeError("Embeddings must be floating-point tensors.")
        if reference_embedding is not None:
            expected = (context_embedding.shape[0], context_embedding.shape[1])
            if (
                reference_embedding.ndim != 3
                or (reference_embedding.shape[0], reference_embedding.shape[-1])
                != expected
                or reference_embedding.shape[1] == 0
            ):
                raise ValueError(
                    "reference embeddings must have shape (batch, samples, features) with at least one sample."
                )
            pos_similarity = self._kernel_call(
                context_embedding, target_embedding, kernel_scale
            ).unsqueeze(1)
            neg_similarity = self._kernel_call(
                context_embedding.unsqueeze(1), reference_embedding, kernel_scale
            )
        else:
            if context_embedding.shape[0] < 2 and not self.pos_as_neg:
                raise ValueError(
                    "In-batch references require at least two pairs; provide explicit references or use drop_last=True."
                )
            similarity = self._kernel_call(
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
        output = self.logits(
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

    def logits(
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
            kernel=self._kernel_call,
            kernel_scale=self.kernel_scale,
        )
        return {
            "pos_similarity": pos_similarity,
            "neg_similarity": neg_similarity,
            "log_baseline": log_baseline,
            "pos_logits": pos_similarity - log_baseline,
            "neg_logits": neg_similarity - log_baseline,
        }
