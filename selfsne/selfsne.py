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
        baseline=None,
        projector=nn.Identity(),
        prior=MixturePrior(num_dims=2, num_components=1),
        similarity_loss=NCE("studentt"),
        redundancy_loss=RedundancyReduction(2),
        similarity_weight=1.0,
        redundancy_weight=1.0,
        rate_weight=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        projector_weight_decay=0.0,
        lr_scheduler=False,
        lr_warmup_steps=5,
        target_lr_steps=5,
        lr_decay_rate=0.95,
    ):
        self.kwargs = locals()
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.pair_sampler = pair_sampler
        self.prior = prior
        self.baseline = baseline
        self.similarity_loss = similarity_loss
        self.redundancy_loss = redundancy_loss
        self.steps_before_lr_decay = lr_warmup_steps + target_lr_steps

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "projector",
                "pair_sampler",
                "prior",
                "baseline",
                "similarity_loss",
                "redundancy_loss",
            ],
        )

    def forward(self, batch):
        return self.projector(self.encoder(batch))

    def loss(self, batch, batch_idx, mode=""):
        x, y = self.pair_sampler(batch)

        x = self(x)
        y = self(y)

        sg_y = stop_gradient(y)

        similarity = self.similarity_loss(
            x, y, log_baseline=self.baseline(sg_y) if self.baseline is not None else 0
        )
        redundancy = self.redundancy_loss(x, y)
        prior_log_prob = -self.prior.log_prob(sg_y).mean()
        rate = -self.prior.rate(y).mean()

        loss = {
            mode + "similarity": similarity,
            mode + "redundancy": redundancy,
            mode + "rate": rate,
            mode + "prior_entropy": self.prior.entropy(),
            mode
            + "loss": (
                prior_log_prob
                + rate * self.hparams.rate_weight
                + similarity * self.hparams.similarity_weight
                + redundancy * self.hparams.redundancy_weight
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
                "weight_decay": self.hparams.projector_weight_decay,
            },
            {
                "params": self.prior.parameters(),
                "weight_decay": 0.0,
                "lr": self.prior.hparams.lr,
            },
        ]
        if self.baseline is not None:
            params_list.append({"params": self.baseline.parameters()})

        optimizer = optim.AdamW(
            params_list,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_scheduler:

            def lr_lambda(step):
                # linear warmup
                if step < self.hparams.lr_warmup_steps:
                    lr_scale = step / self.hparams.lr_warmup_steps
                elif step < self.steps_before_lr_decay:
                    lr_scale = 1
                # exponential decay
                else:
                    lr_scale = self.hparams.lr_decay_rate ** (
                        step - self.steps_before_lr_decay
                    )

                return lr_scale

            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda, verbose=False
            )

            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "loss",
                }
            ]
        else:
            return optimizer

    def load_from_checkpoint(self, *args, **kwargs):
        return super().load_from_checkpoint(*args, **kwargs, **self.kwargs)


SelfSNE.load_from_checkpoint.__doc__ = pl.LightningModule.load_from_checkpoint.__doc__
