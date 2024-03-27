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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import pytorch_lightning as pl

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from selfsne.utils import stop_gradient


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    target_steps: int,
    cosine_steps: int,
) -> lr_scheduler.LRScheduler:
    """
    Returns a learning rate scheduler based on input steps and optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for which the scheduler is to be created.
        warmup_steps (int): Number of warmup steps.
        target_steps (int): Number of target steps.
        cosine_steps (int): Number of cosine annealing steps.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler.

    Example:
        >>> optimizer = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        >>> warmup_steps = 1000
        >>> target_steps = 5000
        >>> cosine_steps = 5000
        >>> scheduler = get_lr_scheduler(optimizer, warmup_steps, target_steps, cosine_steps)
    """
    if warmup_steps + target_steps + cosine_steps == 0:
        scheduler = lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        milestones = np.cumsum([warmup_steps, target_steps])

        linear_warmup = (
            lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, total_iters=warmup_steps
            )
            if warmup_steps > 0
            else lr_scheduler.ConstantLR(optimizer, factor=1.0)
        )
        target_lr = lr_scheduler.ConstantLR(optimizer, factor=1.0)
        cosine_annealing = (
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0)
            if cosine_steps > 0
            else lr_scheduler.ConstantLR(optimizer, factor=1.0)
        )

        scheduler = lr_scheduler.SequentialLR(
            optimizer, [linear_warmup, target_lr, cosine_annealing], milestones
        )

    return scheduler


def visualize_lr_schedule(
    warmup_steps: int, target_steps: int, cosine_steps: int, lr: float = 1.0
) -> Figure:
    """
    Visualizes learning rate schedule and returns a matplotlib figure.

    Args:
        warmup_steps (int): Number of warmup steps in the learning rate scheduler.
        target_steps (int): Number of target steps in the learning rate scheduler.
        cosine_steps (int): Number of cosine steps in the learning rate scheduler.
        lr (float, optional): Initial learning rate for the SGD optimizer. Defaults to 1.0.

    Returns:
        matplotlib.figure.Figure: The figure object of the plotted learning rate schedule.

    Example:
        >>> warmup_steps = 1000
        >>> target_steps = 5000
        >>> cosine_steps = 5000
        >>> fig = visualize_lr_schedule(warmup_steps, target_steps, cosine_steps, lr=1.0)
        >>> plt.show()
    """
    # Initialize parameters
    params = [torch.randn(1, requires_grad=True)]

    # Set up SGD optimizer with specified learning rate
    optimizer = optim.SGD(params, lr=lr)

    # Get learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, warmup_steps, target_steps, cosine_steps)

    # Calculate total steps
    total_steps = warmup_steps + target_steps + cosine_steps

    # Track learning rate values
    lr_values = []
    for step in range(total_steps):
        lr = optimizer.param_groups[0]["lr"]
        lr_values.append(lr)
        scheduler.step()

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(range(total_steps), lr_values)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    return fig


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
        betas=(0.9, 0.95),
        dampening=0,
        nesterov=False,
        weight_decay=0.0,
        encoder_weight_decay=None,
        projector_weight_decay=None,
        encoder_x_weight_decay=None,
        projector_x_weight_decay=None,
        decoder_weight_decay=None,
        prior_weight_decay=None,
        lr_scheduler=False,
        lr_warmup_steps=0,
        lr_target_steps=0,
        lr_cosine_steps=0,
        concat_chunk_encode=True,
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
        self.hparams.encoder_weight_decay = (
            encoder_weight_decay if encoder_weight_decay is not None else weight_decay
        )
        self.hparams.projector_weight_decay = (
            projector_weight_decay
            if projector_weight_decay is not None
            else weight_decay
        )
        self.hparams.encoder_x_weight_decay = (
            encoder_x_weight_decay
            if encoder_x_weight_decay is not None
            else weight_decay
        )
        self.hparams.projector_x_weight_decay = (
            projector_x_weight_decay
            if projector_x_weight_decay is not None
            else weight_decay
        )
        self.hparams.decoder_weight_decay = (
            decoder_weight_decay if decoder_weight_decay is not None else weight_decay
        )
        self.hparams.prior_weight_decay = (
            prior_weight_decay if prior_weight_decay is not None else weight_decay
        )

    def forward(self, batch):
        return self.projector(self.encoder(batch))

    def loss(self, batch, batch_idx, mode=""):
        if self.pair_sampler is not None:
            x, y = self.pair_sampler(batch)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = batch

        if self.encoder_x is not None:
            h_x = self.encoder_x(x)
            z_x = self.projector_x(h_x)
            h_y = self.encoder(y)
            z_y = self.projector(h_y)
        else:
            if self.hparams.concat_chunk_encode:
                h_x, h_y = torch.chunk(self.encoder(torch.cat([x, y])), 2)
                z_x, z_y = torch.chunk(self.projector(torch.cat([h_x, h_y])), 2)
            else:
                h_x = self.encoder(x)
                h_y = self.encoder(y)
                z_x = self.projector(h_x)
                z_y = self.projector(h_y)

        if self.similarity_loss is not None:
            (
                pos_logits,
                neg_logits,
                kld,
                rkld,
                jsd,
                pos_prob,
                neg_prob,
                accuracy,
                recall,
                precision,
                spec,
                npv,
                log_baseline,
                similarity,
            ) = self.similarity_loss(z_x=z_x, z_y=z_y, h_x=h_x, h_y=h_y, x=x, y=y)
            self.log(mode + "similarity", similarity.item(), prog_bar=True)
            self.log(mode + "pos_logits", pos_logits.item(), prog_bar=True)
            self.log(mode + "neg_logits", neg_logits.item(), prog_bar=True)
            self.log(mode + "kld", kld.item(), prog_bar=True)
            self.log(mode + "rkld", rkld.item(), prog_bar=True)
            self.log(mode + "jsd", jsd.item(), prog_bar=True)
            self.log(mode + "log_baseline", log_baseline.item(), prog_bar=True)
            self.log(mode + "pos_prob", pos_prob.item(), prog_bar=True)
            self.log(mode + "neg_prob", neg_prob.item(), prog_bar=True)
            self.log(mode + "accuracy", accuracy.item(), prog_bar=True)
            self.log(mode + "recall", recall.item(), prog_bar=True)
            self.log(mode + "precision", precision.item(), prog_bar=True)
            self.log(mode + "specificity", spec.item(), prog_bar=True)
            self.log(mode + "npv", npv.item(), prog_bar=True)

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
            if hasattr(self.prior, "entropy"):
                self.log(
                    mode + "prior_entropy", self.prior.entropy().item(), prog_bar=True
                )
            if hasattr(self.prior, "mixture") and hasattr(
                self.prior.mixture, "entropy"
            ):
                self.log(
                    mode + "cluster_perplexity",
                    self.prior.mixture.entropy().exp().item(),
                    prog_bar=True,
                )
            self.log(mode + "rate_weight", float(self.rate_weight), prog_bar=True)

        loss = (
            (prior_log_prob if self.prior is not None else 0)
            + ((rate * self.rate_weight) if self.prior is not None else 0)
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
        self.log(mode + "loss", loss.item(), prog_bar=True)

        return loss

    @property
    def rate_weight(self):
        if self.hparams.rate_start_step + self.hparams.rate_warmup_steps == 0:
            rate_weight = self.hparams.rate_weight
        elif (
            self.hparams.rate_start_step
            <= self.current_epoch
            < self.hparams.rate_start_step + self.hparams.rate_warmup_steps
        ):
            rate_weight = self.hparams.rate_weight * (
                (self.current_epoch - self.hparams.rate_start_step + 1)
                / self.hparams.rate_warmup_steps
            )
        elif self.current_epoch >= self.hparams.rate_start_step:
            rate_weight = self.hparams.rate_weight
        else:
            rate_weight = 0
        return rate_weight

    def training_step(self, batch, batch_idx):
        return self.loss(batch, batch_idx, mode="")

    def validation_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="val_")

    def test_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="test_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction = {}

        prediction["embedding"] = self(batch)
        if self.prior is not None and hasattr(self.prior, "assign_labels"):
            prediction["labels"] = self.prior.assign_labels(prediction["embedding"])

        return prediction

    def configure_optimizers(self):
        params_list = [
            {
                "params": self.encoder.parameters(),
                "weight_decay": self.hparams.encoder_weight_decay,
            },
            {
                "params": self.projector.parameters(),
                "weight_decay": self.hparams.projector_weight_decay,
            },
        ]
        if self.encoder_x is not None:
            params_list.append(
                {
                    "params": self.encoder_x.parameters(),
                    "weight_decay": self.hparams.encoder_x_weight_decay,
                },
            )
            params_list.append(
                {
                    "params": self.projector_x.parameters(),
                    "weight_decay": self.hparams.projector_x_weight_decay,
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
                    "weight_decay": self.hparams.prior_weight_decay,
                    "lr": self.prior.hparams.lr,
                }
            )
        if self.hparams.optimizer.lower() == "adam":
            optimizer = optim.AdamW(
                params_list,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=self.hparams.betas,
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
            scheduler = get_lr_scheduler(
                optimizer,
                warmup_steps=self.hparams.lr_warmup_steps,
                target_steps=self.hparams.lr_target_steps,
                cosine_steps=self.hparams.lr_cosine_steps,
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
