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
        encoder_x=None,
        projector_x=nn.Identity(),
        pair_sampler=None,
        projector=nn.Identity(),
        decoder=nn.Identity(),
        prior=None,
        similarity_loss=None,
        redundancy_loss=None,
        reconstruction_loss=None,
        similarity_weight=1.0,
        redundancy_weight=1.0,
        rate_weight=1.0,
        rate_start_step=0,
        rate_warmup_steps=0,
        learning_rate=1e-3,
        optimizer="adam",
        momentum=0,
        dampening=0,
        nesterov=False,
        weight_decay=0.0,
        projector_weight_decay=0.0,
        decoder_weight_decay=0.0,
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
        self.encoder_x = encoder_x
        self.projector_x = projector_x
        self.decoder = decoder
        self.pair_sampler = pair_sampler
        self.prior = prior
        self.similarity_loss = similarity_loss
        self.redundancy_loss = redundancy_loss
        self.reconstruction_loss = reconstruction_loss
        self.save_hyperparameters(
            ignore=[
                "encoder",
                "projector",
                "encoder_x",
                "projector_x",
                "decoder",
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

        if self.pair_sampler is not None:
            x, y = self.pair_sampler(batch)
        else:
            x = batch
            y = batch

        if self.encoder_x is not None:
            h_x = self.encoder_x(x)
            z_x = self.projector_x(h_x)
        else:
            h_x = self.encoder(x)
            z_x = self.projector(h_x)

        h_y = self.encoder(y)
        z_y = self.projector(h_y)

        if self.similarity_loss is not None:
            (
                pos_log_density_ratio,
                neg_log_density_ratio,
                log_baseline,
                similarity,
            ) = self.similarity_loss(z_x, z_y, y)
            self.log(mode + "similarity", similarity.item(), prog_bar=True)
            self.log(
                mode + "pos_log_density_ratio", pos_log_density_ratio.item(), prog_bar=True
            )
            self.log(
                mode + "neg_log_density_ratio", neg_log_density_ratio.item(), prog_bar=True
            )
            self.log(mode + "log_baseline", log_baseline.item(), prog_bar=True)

        if self.redundancy_loss is not None:
            redundancy = self.redundancy_loss(z_x, z_y)
            self.log(mode + "redundancy", redundancy.item(), prog_bar=True)

        if self.reconstruction_loss is not None:
            y_hat = self.decoder(stop_gradient(z_y))
            reconstruction = self.reconstruction_loss(y_hat, y).mean()
            self.log(mode + "reconstruction", reconstruction.item(), prog_bar=True)

        if self.prior is not None:
            prior_log_prob = -self.prior.log_prob(stop_gradient(z_y)).mean()
            rate = -self.prior.rate(z_y).mean()
            self.log(mode + "rate", rate.item(), prog_bar=True)
            self.log(
                mode + "prior_entropy",
                self.prior.entropy().item(),
                prog_bar=True,
            )
            self.log(
                mode + "cluster_perplexity",
                self.prior.mixture.entropy().exp().item(),
                prog_bar=True,
            )
            if (
                self.hparams.rate_start_step
                <= self.current_epoch
                < self.hparams.rate_start_step + self.hparams.rate_warmup_steps
            ):
                rate_weight = self.hparams.rate_weight * (
                    (self.current_epoch - self.hparams.rate_start_step + 1)
                    / self.hparams.rate_warmup_steps
                )
            elif self.current_epoch > self.hparams.rate_start_step:
                rate_weight = self.hparams.rate_weight
            else:
                rate_weight = 0
            self.log(mode + "rate_weight", float(rate_weight), prog_bar=True)

        loss = (
            (prior_log_prob if self.prior is not None else 0)
            + ((rate * rate_weight) if self.prior is not None else 0)
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
            + (reconstruction if self.reconstruction_loss is not None else 0)
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(batch, batch_idx, mode="")

    def validation_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="val_")

    def test_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="test_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction = {}

        prediction["embedding"] = self(batch)
        if self.prior is not None:
            prediction["labels"] = self.prior.assign_labels(prediction["embedding"])

        return prediction

    def configure_optimizers(self):
        params_list = [
            {"params": self.encoder.parameters()},
            {
                "params": self.projector.parameters(),
                "weight_decay": self.hparams.projector_weight_decay,
            },
        ]
        if self.encoder_x is not None:
            params_list.append({"params": self.encoder_x.parameters()})
            params_list.append(
                {
                    "params": self.projector_x.parameters(),
                    "weight_decay": self.hparams.projector_weight_decay,
                }
            )

        if self.pair_sampler is not None:
            params_list.append({"params": self.pair_sampler.parameters()})

        if self.similarity_loss is not None:
            params_list.append({"params": self.similarity_loss.parameters()})

        if self.redundancy_loss is not None:
            params_list.append({"params": self.redundancy_loss.parameters()})

        if self.reconstruction_loss is not None:
            params_list.append(
                {
                    "params": self.decoder.parameters(),
                    "weight_decay": self.hparams.decoder_weight_decay,
                }
            )

        if self.prior is not None:
            params_list.append(
                {
                    "params": self.prior.parameters(),
                    "weight_decay": 0.0,
                    "lr": self.prior.hparams.lr,
                }
            )
        if self.hparams.optimizer.lower() == "adam":
            optimizer = optim.AdamW(
                params_list,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                params_list,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
                dampening=self.hparams.dampening,
                nesterov=self.hparams.nesterov,
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
