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
from einops import rearrange
from typing import Optional, Union, Tuple, Dict


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
    def __init__(
        self,
        loss,
        target_encoder=None,
        context_encoder=None,
        target_tokenizer=None,
        context_tokenizer=None,
        embedding_head: Optional[torch.nn.Module] = None,
        baseline_head: Optional[torch.nn.Module] = None,
        learning_rate=3e-4,
        optimizer="adam",
        momentum=0,
        betas=(0.9, 0.95),
        dampening=0,
        nesterov=False,
        weight_decay=0.0,
        lr_scheduler: bool = False,
        lr_warmup_steps=0,
        lr_target_steps=0,
        lr_cosine_steps=0,
    ):
        super().__init__()
        if loss is None:
            raise ValueError("A loss module must be provided.")
        self.loss = loss

        if target_encoder is None and context_encoder is None:
            raise ValueError(
                "At least one of target_encoder or context_encoder must be provided."
            )
        if target_tokenizer is None and context_tokenizer is None:
            raise ValueError(
                "At least one of target_tokenizer or context_tokenizer must be provided."
            )
        if target_encoder is None:
            target_encoder = context_encoder
        if context_encoder is None:
            context_encoder = target_encoder
        if target_tokenizer is None:
            target_tokenizer = context_tokenizer
        if context_tokenizer is None:
            context_tokenizer = target_tokenizer

        self.target_encoder = target_encoder
        self.context_encoder = context_encoder
        self.target_tokenizer = target_tokenizer
        self.context_tokenizer = context_tokenizer

        self.embedding_head = embedding_head
        self.baseline_head = baseline_head

        self.save_hyperparameters(
            ignore=[
                "loss",
                "target_encoder",
                "context_encoder",
                "target_tokenizer",
                "context_tokenizer",
                "embedding_head",
                "baseline_head",
            ]
        )
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer.lower()
        self.momentum = momentum
        self.betas = betas
        self.dampening = dampening
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.lr_sched_flag = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_target_steps = lr_target_steps
        self.lr_cosine_steps = lr_cosine_steps

    def process_batch(self, batch) -> Tuple:
        if isinstance(batch, dict):
            context = batch.get("context")
            target = batch.get("target")
            reference = batch.get("reference", None)
            if context is None or target is None:
                raise ValueError(
                    "Dictionary batch must have at least 'context' and 'target' keys."
                )
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                context, target = batch
                reference = None
            elif len(batch) == 3:
                context, target, reference = batch
            else:
                raise ValueError("Batch must contain 2 or 3 items.")
        else:
            context = target = batch
            reference = None
        return context, target, reference

    def get_embeddings(self, context, target, reference=None) -> Tuple:
        context_tokens = self.context_tokenizer(context)
        target_tokens = self.target_tokenizer(target)
        context_encoding = self.context_encoder(context_tokens)
        target_encoding = self.target_encoder(target_tokens)
        baseline_embedding = None
        if self.baseline_head is not None:
            baseline_embedding = self.baseline_head(context_encoding)
        if self.embedding_head is not None:
            context_embedding = self.embedding_head(context_encoding)
            target_embedding = self.embedding_head(target_encoding)
        else:
            context_embedding = context_encoding
            target_embedding = target_encoding
        if reference is not None:
            reference_tokens = self.target_tokenizer(
                rearrange(reference, "b s ... -> (b s) ...")
            )
            reference_encoding = self.target_encoder(reference_tokens)
            if self.embedding_head is not None:
                reference_embedding = self.embedding_head(reference_encoding)
            else:
                reference_embedding = reference_encoding
            num_samples = reference.shape[1]
            reference_embedding = rearrange(
                reference_embedding, "(b s) ... -> b s ...", s=num_samples
            )
        else:
            reference_embedding = None
        return (
            context_embedding,
            target_embedding,
            reference_embedding,
            baseline_embedding,
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        context, target, reference = self.process_batch(batch)
        (
            context_embedding,
            target_embedding,
            reference_embedding,
            baseline_embedding,
        ) = self.get_embeddings(context, target, reference)
        output = {
            "context_embedding": context_embedding,
            "target_embedding": target_embedding,
        }
        if baseline_embedding is not None:
            output["baseline_embedding"] = baseline_embedding
        if reference_embedding is not None:
            output["reference_embedding"] = reference_embedding
        return output

    def compute_loss(self, batch, batch_idx, mode=""):
        context, target, reference = self.process_batch(batch)
        (
            context_embedding,
            target_embedding,
            reference_embedding,
            baseline_embedding,
        ) = self.get_embeddings(context, target, reference)
        loss_dict = self.loss(
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            baseline_embedding=baseline_embedding,
            reference_embedding=reference_embedding,
        )
        for key, value in loss_dict.items():
            self.log(f"{mode}{key}", value.item(), prog_bar=True)
        return loss_dict["loss"]

    def predict_sample_logits(self, batch) -> torch.Tensor:
        context, target, reference = self.process_batch(batch)
        (
            context_embedding,
            target_embedding,
            reference_embedding,
            baseline_embedding,
        ) = self.get_embeddings(context, target, reference)
        pos_logits, neg_logits = self.loss.logits(
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            kernel_scale=self.loss.kernel_scale,
            reference_embedding=reference_embedding,
        )
        log_baseline = self.loss.baseline(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            context_embedding=context_embedding,
            target_embedding=target_embedding,
            baseline_embedding=baseline_embedding,
            reference_embedding=reference_embedding,
        )
        samplewise_pos_logits = pos_logits - log_baseline
        return samplewise_pos_logits

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="")

    def validation_step(self, batch, batch_idx):
        self.compute_loss(batch, batch_idx, mode="val_")

    def test_step(self, batch, batch_idx):
        self.compute_loss(batch, batch_idx, mode="test_")

    def predict_step(
        self, batch, batch_idx, dataloader_idx=0
    ) -> Dict[str, torch.Tensor]:
        context, target, _ = self.process_batch(batch)
        target_tokens = self.target_tokenizer(target)
        return {"embedding": self.target_encoder(target_tokens)}

    def configure_optimizers(self):
        modules = []
        for module in [
            self.target_encoder,
            self.context_encoder,
            self.target_tokenizer,
            self.context_tokenizer,
        ]:
            if module is not None and module not in modules:
                modules.append(module)
        params_list = [{"params": m.parameters()} for m in modules]
        params_list.append({"params": self.loss.parameters()})
        if self.baseline_head is not None:
            params_list.append({"params": self.baseline_head.parameters()})
        if self.embedding_head is not None:
            params_list.append({"params": self.embedding_head.parameters()})
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
