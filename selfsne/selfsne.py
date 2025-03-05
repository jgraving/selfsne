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

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import pytorch_lightning as pl
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
                optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps
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
        pair_sampler=PairSampler(),
        projector=nn.Identity(),
        similarity_loss=None,
        learning_rate=3e-4,
        optimizer="adam",
        momentum=0,
        betas=(0.9, 0.95),
        dampening=0,
        nesterov=False,
        weight_decay=0.0,
        lr_scheduler=False,
        lr_warmup_steps=0,
        lr_target_steps=0,
        lr_cosine_steps=0,
    ):
        super().__init__()
        if target_encoder is None and context_encoder is None:
            raise ValueError(
                "At least one of target_encoder or context_encoder must be provided."
            )
        # If one encoder is missing, use the other.
        if target_encoder is None:
            target_encoder = context_encoder
        if context_encoder is None:
            context_encoder = target_encoder

        self.target_encoder = target_encoder
        self.context_encoder = context_encoder
        self.similarity_loss = similarity_loss
        self.save_hyperparameters(
            ignore=["target_encoder", "context_encoder", "similarity_loss"]
        )

    def forward(self, batch):
        # For inference, use the target_encoder.
        return self.target_encoder(batch)

    def loss(self, batch, batch_idx, mode=""):
        # Assume batch is a tuple (context, target)
        context, target = batch
        if self.similarity_loss is not None:
            loss_dict = self.similarity_loss(
                context=context,
                target=target,
                target_encoder=self.target_encoder,
                context_encoder=self.context_encoder,
            )
        else:
            loss_dict = {"loss": torch.tensor(0.0)}
        for key, value in loss_dict.items():
            self.log(f"{mode}{key}", value.item(), prog_bar=True)
        return loss_dict["loss"]

    def training_step(self, batch, batch_idx):
        return self.loss(batch, batch_idx, mode="")

    def validation_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="val_")

    def test_step(self, batch, batch_idx):
        self.loss(batch, batch_idx, mode="test_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return {"embedding": self(batch)}

    def configure_optimizers(self):
        params_list = []
        # Avoid duplicating parameters if both encoders are the same object.
        if self.target_encoder is self.context_encoder:
            params_list.append(
                {
                    "params": self.target_encoder.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                }
            )
        else:
            for model in [self.target_encoder, self.context_encoder]:
                params_list.append(
                    {
                        "params": model.parameters(),
                        "weight_decay": self.hparams.weight_decay,
                    }
                )
        if self.similarity_loss is not None:
            params_list.append({"params": self.similarity_loss.parameters()})

        if self.hparams.optimizer.lower() == "adam":
            opt = optim.AdamW(
                params_list,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=self.hparams.betas,
            )
        elif self.hparams.optimizer.lower() == "sgd":
            opt = optim.SGD(
                params_list,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
                dampening=self.hparams.dampening,
                nesterov=self.hparams.nesterov,
            )
        else:
            raise ValueError("Unsupported optimizer type. Use 'adam' or 'sgd'.")

        if self.hparams.lr_scheduler:
            sched = get_lr_scheduler(
                opt,
                warmup_steps=self.hparams.lr_warmup_steps,
                target_steps=self.hparams.lr_target_steps,
                cosine_steps=self.hparams.lr_cosine_steps,
            )
            return [opt], [
                {
                    "scheduler": sched,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "loss",
                }
            ]
        else:
            return opt
