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

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np

from selfsne.utils import stop_gradient


class SelfSNE(pl.LightningModule):
    """Self-Supervised Noise Embedding"""

    def __init__(
        self,
        encoder,
        pair_sampler=None,
        log_baseline=None,
        projector=nn.Identity(),
        prior=None,
        similarity_loss=None,
        redundancy_loss=None,
        similarity_weight=1.0,
        redundancy_weight=1.0,
        rate_weight=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        projector_weight_decay=0.0,
        lr_scheduler=False,
        lr_warmup_steps=10,
        lr_target_steps=10,
        lr_cosine_steps=30,
        lr_cosine_steps_per_cycle=10,
        lr_warm_restarts=False,
        lr_decay_rate=0.9,
    ):
        self.kwargs = locals()
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.pair_sampler = pair_sampler
        self.prior = prior
        self.log_baseline = log_baseline
        self.similarity_loss = similarity_loss
        self.redundancy_loss = redundancy_loss
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

        loss = {}

        if self.pair_sampler is not None:
            x, y = self.pair_sampler(batch)
        else:
            x = batch
            y = batch

        x = self(x)
        y = self(y)

        sg_y = stop_gradient(y)

        if self.similarity_loss is not None:
            similarity = self.similarity_loss(
                x,
                y,
                log_baseline=self.log_baseline(sg_y)
                if self.log_baseline is not None
                else 0,
            )
            loss[mode + "similarity"] = similarity

        if self.redundancy_loss is not None:
            redundancy = self.redundancy_loss(x, y)
            loss[mode + "redundancy"] = redundancy

        if self.prior is not None:
            prior_log_prob = -self.prior.log_prob(sg_y).mean()
            rate = -self.prior.rate(y).mean()
            loss[mode + "rate"] = rate
            loss[mode + "prior_entropy"] = self.prior.entropy()

        loss[mode + "loss"] = (
            (prior_log_prob if self.prior is not None else 0)
            + ((rate * self.hparams.rate_weight) if self.prior is not None else 0)
            + (
                (similarity * self.hparams.similarity_weight)
                if self.similarity_loss is not None
                else 0
            )
            + (
                (redundancy * self.hparams.redundancy_weight)
                if self.redundancy_loss is not None
                else 0
            )
        )

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
        prediction = {}

        prediction["embedding"] = self(batch)
        if self.prior is not None:
            prediction["prior_log_prob"] = self.prior.log_prob(prediction["embedding"])
            prediction["labels"] = self.prior.assign_labels(prediction["embedding"])

        return prediction

    def configure_optimizers(self):
        params_list = [
            {"params": self.encoder.parameters()},
            {"params": self.pair_sampler.parameters()},
            {
                "params": self.projector.parameters(),
                "weight_decay": self.hparams.projector_weight_decay,
            },
        ]

        if self.similarity_loss is not None:
            params_list.append({"params": self.similarity_loss.parameters()})

        if self.redundancy_loss is not None:
            params_list.append({"params": self.redundancy_loss.parameters()})

        if self.prior is not None:
            params_list.append(
                {
                    "params": self.prior.parameters(),
                    "weight_decay": 0.0,
                    "lr": self.prior.hparams.lr,
                }
            )

        if self.log_baseline is not None:
            params_list.append({"params": self.log_baseline.parameters()})

        optimizer = optim.AdamW(
            params_list,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler:

            lr_warmup_steps = self.hparams.lr_warmup_steps
            lr_target_steps = self.hparams.lr_target_steps
            lr_cosine_steps = self.hparams.lr_cosine_steps
            lr_cosine_steps_per_cycle = self.hparams.lr_cosine_steps_per_cycle
            lr_decay_rate = self.hparams.lr_decay_rate

            milestones = np.cumsum(
                [
                    lr_warmup_steps,
                    lr_target_steps,
                    lr_cosine_steps + (lr_warmup_steps == 0 and lr_target_steps == 0),
                ]
            )
            linear_warmup = (
                lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-8, total_iters=lr_warmup_steps
                )
                if lr_warmup_steps > 0
                else lr_scheduler.ConstantLR(optimizer, factor=1.0)
            )
            target_lr = lr_scheduler.ConstantLR(optimizer, factor=1.0)
            if self.hparams.lr_warm_restarts:
                cosine_annealing = lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=lr_cosine_steps_per_cycle, eta_min=0
                )
            else:
                cosine_annealing = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=lr_cosine_steps_per_cycle, eta_min=0
                )

            exp_decay = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
            scheduler = lr_scheduler.SequentialLR(
                optimizer,
                [linear_warmup, target_lr, cosine_annealing, exp_decay],
                milestones,
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
