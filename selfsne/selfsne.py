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


import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

import copy

from selfsne.kernels import KERNELS
from selfsne.losses import categorical_infonce, redundancy_reduction
from selfsne.neighbors import Queue
from selfsne.prior import MixturePrior


class SelfSNE(pl.LightningModule):
    """Self-Supervised Noise Embedding"""

    def __init__(
        self,
        encoder,
        augmenter_a=nn.Identity(),
        augmenter_b=nn.Identity(),
        prior=MixturePrior(num_dims=2, num_components=1),
        embedding_dims=2,
        kernel="studentt",
        similarity_multiplier=1.0,
        redundancy_multiplier=1.0,
        rate_multiplier=0.1,
        learning_rate=1e-3,
        weight_decay=0.01,
    ):
        super().__init__()
        self.encoder = encoder
        self.augmenter_a = augmenter_a
        self.augmenter_b = augmenter_b
        self.prior = prior
        self.kernel = KERNELS[kernel]
        self.bn = nn.BatchNorm1d(embedding_dims, affine=False)

        self.save_hyperparameters(
            "embedding_dims",
            "kernel",
            "similarity_multiplier",
            "redundancy_multiplier",
            "rate_multiplier",
            "learning_rate",
            "weight_decay",
        )

    def forward(self, batch):
        return self.encoder(batch)

    def loss(self, batch, mode=""):
        query = self.encoder(self.augmenter_a(batch))
        key = self.encoder(self.augmenter_b(batch))

        similarity = categorical_infonce(query, key, key, self.kernel).mean()

        redundancy = redundancy_reduction(query, key, self.bn).mean()

        rate = -(
            self.prior.log_prob(query.clone().detach()).mean()
            + self.prior.commitment(query).mean() * self.hparams.rate_multiplier
        )

        loss = {
            mode + "similarity": similarity,
            mode + "redundancy": redundancy,
            mode + "rate": rate,
            mode + "prior_entropy": self.prior.entropy(),
            mode + "unweighted_loss": rate + similarity + redundancy,
            mode
            + "loss": (
                rate
                + similarity * self.hparams.similarity_multiplier
                + redundancy * self.hparams.redundancy_multiplier
            ),
        }
        for key in loss.keys():
            self.log(key, loss[key], prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(batch, mode="")["loss"]

    def validation_step(self, batch, batch_idx):
        self.loss(batch, mode="val_")

    def test_step(self, batch, batch_idx):
        self.loss(batch, mode="test_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        embedded = self(batch)
        prior_log_prob = self.prior.log_prob(embedded)
        labels = self.prior.assign_labels(embedded)

        return {
            "embedding": embedded.cpu().numpy(),
            "labels": labels.cpu().numpy(),
            "prior_log_prob": prior_log_prob.cpu().numpy(),
        }

    def configure_optimizers(self):
        params_list = [
            {"params": self.encoder.parameters()},
            {"params": self.bn.parameters()},
        ]
        params_list.append(
            {"params": self.prior.parameters(), "weight_decay": 0.0, "lr": 0.1}
        )

        return optim.AdamW(
            params_list,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
