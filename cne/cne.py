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
from cne.kernels import KERNELS
from cne.losses import infonce, redundancy_reduction


class CNE(pl.LightningModule):
    """Contrastive Noise Embedding"""

    def __init__(
        self,
        encoder=None,
        prior=None,
        z_dim=2,
        kernel="studentt",
        similarity_multiplier=1.0,
        redundancy_multiplier=1.0,
        rate_multiplier=1.0,
        learning_rate=1e-3,
        weight_decay=0.01,
    ):
        super().__init__()
        self.encoder = encoder
        self.prior = prior

        self.kernel = KERNELS[kernel]
        self.bn = nn.BatchNorm1d(z_dim, affine=False)
        self.save_hyperparameters(
            "z_dim",
            "kernel",
            "similarity_multiplier",
            "redundancy_multiplier",
            "rate_multiplier",
            "learning_rate",
            "weight_decay",
        )

    def forward(self, batch):
        return self.encoder(batch)

    def loss(self, z_a, z_b, mode=""):
        similarity = infonce(z_a, z_b, self.kernel).mean()
        redundancy = redundancy_reduction(z_a, z_b, self.bn).mean()
        rate = -self.prior.log_prob(z_a).mean()
        loss = {
            mode + "similarity": similarity,
            mode + "redundancy": redundancy,
            mode + "rate": rate,
            mode + "prior_entropy": self.prior.entropy(),
            mode + "unweighted_loss": rate + similarity + redundancy,
            mode
            + "loss": (
                rate * self.hparams.rate_multiplier
                + similarity * self.hparams.similarity_multiplier
                + redundancy * self.hparams.redundancy_multiplier
            ),
        }
        for key in loss.keys():
            self.log(key, loss[key], prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(*self(batch), mode="")["loss"]

    def validation_step(self, batch, batch_idx):
        self.loss(*self(batch), mode="val_")

    def test_step(self, batch, batch_idx):
        self.loss(*self(batch), mode="test_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        embedded_a, embedded_b = self(batch)
        prior_log_prob_a = self.prior.log_prob(embedded_a)
        prior_log_prob_b = self.prior.log_prob(embedded_b)
        labels_a = self.prior.assign_labels(embedded_a)
        labels_b = self.prior.assign_labels(embedded_b)

        return {
            "embedding_a": embedded_a.cpu().numpy(),
            "embedding_b": embedded_b.cpu().numpy(),
            "labels_a": labels_a.cpu().numpy(),
            "labels_a": labels_a.cpu().numpy(),
            "prior_log_prob_a": prior_log_prob_a.cpu().numpy(),
            "prior_log_prob_b": prior_log_prob_b.cpu().numpy(),
        }

    def configure_optimizers(self):
        params_list = [
            {"params": self.encoder.parameters()},
            {"params": self.bn.parameters()},
        ]
        params_list.append({"params": self.prior.parameters(), "weight_decay": 0.0})

        return optim.AdamW(
            params_list,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
