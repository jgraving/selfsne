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

from selfsne.prior import MixturePrior
from selfsne.losses import NCE, RedundancyReduction
from selfsne.neighbors import NearestNeighborSampler
from selfsne.utils import stop_gradient


class SelfSNE(pl.LightningModule):
    """Self-Supervised Noise Embedding"""

    def __init__(
        self,
        encoder,
        pair_sampler,
        projector=nn.Identity(),
        prior=MixturePrior(num_dims=2, num_components=1),
        similarity_loss=NCE("studentt"),
        redundancy_loss=RedundancyReduction(2),
        similarity_multiplier=1.0,
        redundancy_multiplier=1.0,
        rate_multiplier=0.1,
        learning_rate=1e-3,
        weight_decay=0.01,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.pair_sampler = pair_sampler
        self.prior = prior
        self.similarity_loss = similarity_loss
        self.redundancy_loss = redundancy_loss

        self.save_hyperparameters(
            "similarity_multiplier",
            "redundancy_multiplier",
            "rate_multiplier",
            "learning_rate",
            "weight_decay",
            ignore=[
                "encoder",
                "projector",
                "pair_sampler",
                "prior",
                "similarity_loss",
                "redundancy_loss",
            ],
        )

    def forward(self, batch):
        return self.projector(self.encoder(batch))

    def loss(self, batch, batch_idx, mode=""):
        query, key = self.pair_sampler(batch)

        query = self(query)
        key = self(key)

        similarity = self.similarity_loss(query, key).mean()
        redundancy = self.redundancy_loss(query, key).mean()
        prior_log_prob = -self.prior.log_prob(stop_gradient(query)).mean()
        rate = -self.prior.rate(query).mean()

        loss = {
            mode + "similarity": similarity,
            mode + "redundancy": redundancy,
            mode + "rate": rate,
            mode + "prior_entropy": self.prior.entropy(),
            mode
            + "loss": (
                prior_log_prob
                + rate * self.hparams.rate_multiplier
                + similarity * self.hparams.similarity_multiplier
                + redundancy * self.hparams.redundancy_multiplier
            ),
        }
        for key in loss.keys():
            self.log(key, loss[key], prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(batch, batch_idx, mode="")["loss"]

    def validation_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="val_")

    def test_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="test_")

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
            {"params": self.pair_sampler.parameters()},
            {"params": self.similarity_loss.parameters()},
            {"params": self.redundancy_loss.parameters()},
            {
                "params": self.projector.parameters(),
                "weight_decay": 0.0,
            },
            {
                "params": self.prior.parameters(),
                "weight_decay": 0.0,
                "lr": self.prior.hparams.lr,
            },
        ]

        return optim.AdamW(
            params_list,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
