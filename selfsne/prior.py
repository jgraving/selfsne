# -*- coding: utf-8 -*-
# Copyright 2020 Jacob M. Graving <jgraving@gmail.com>
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
import torch.distributions as D
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

from selfsne.kernels import ROWWISE_KERNELS
from selfsne.baselines import BASELINES
from selfsne.utils import disable_grad, enable_grad, stop_gradient, inverse_softplus


class NormalPrior(pl.LightningModule):
    def __init__(
        self,
        num_features=2,
        lr=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.log_scale = nn.Parameter(inverse_softplus(torch.ones((1, num_features))))
        self.register_buffer("loc", torch.zeros((1, num_features)))

    def log_prob(self, x):
        return D.Normal(self.loc, F.softplus(self.log_scale)).log_prob(x)

    def rate(self, x):
        disable_grad(self)
        rate = self.log_prob(x)
        enable_grad(self)
        return rate


class MixturePrior(pl.LightningModule):
    def __init__(
        self,
        num_features=2,
        num_components=2048,
        kernel="normal",
        logits="learn",
        kernel_scale=1.0,
        baseline=0,
        lr=1.0,
        scheduler_kwargs=None,
    ):
        super().__init__()
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        self.save_hyperparameters()
        if logits not in {"learn", "maxent"}:
            raise ValueError("logits must be 'learn' or 'maxent'.")

        if isinstance(kernel, str):
            self.kernel = ROWWISE_KERNELS[kernel]
        else:
            self.kernel = kernel

        if num_components == 1:
            locs = torch.zeros((num_components, num_features))
            self.register_buffer("locs", locs)
        else:
            self.locs = nn.Parameter(torch.Tensor(num_components, num_features))
            init.normal_(self.locs)

        if logits == "learn":
            self.logits = nn.Parameter(torch.zeros((num_components,)))
            init.zeros_(self.logits)
        elif logits == "maxent":
            logits = torch.zeros((num_components,))
            self.register_buffer("logits", logits)

        if isinstance(baseline, str):
            self.baseline = BASELINES[baseline]()
        else:
            self.baseline = BASELINES["constant"](baseline)

        self.watershed_locs = nn.Parameter(stop_gradient(self.locs))
        self.register_buffer("watershed_assignments", self.watershed_labels())

    @property
    def multinomial(self):
        return D.Multinomial(logits=self.logits)

    @property
    def mixture(self):
        return D.Categorical(logits=self.logits)

    def entropy(self):
        return -(self.mixture.probs * self._log_prob(self.locs)).sum()

    def weighted_log_prob(self, x):
        return self.kernel(
            self.locs.unsqueeze(1), x, self.hparams.kernel_scale
        ) + self.mixture.logits.unsqueeze(1)

    def _log_prob(self, x):
        return self.weighted_log_prob(x).logsumexp(0)

    def log_prob(self, x):
        log_prob = self._log_prob(x)
        return log_prob.unsqueeze(1) - self.baseline(
            pos_similarity=log_prob.unsqueeze(1),
            neg_similarity=log_prob.unsqueeze(0).expand(log_prob.shape[0], -1),
            context_embedding=x,
        )

    def rate(self, x):
        disable_grad(self)
        rate = self.log_prob(x)
        enable_grad(self)
        return rate

    def assign_modes(self, x):
        return self.weighted_log_prob(x).argmax(0)

    def quantize(self, x):
        return self.locs[self.assign_modes(x)]

    def entropy_upper_bound(self):
        return -(self.mixture.probs * self._log_prob(self.watershed_locs)).sum()

    def watershed_labels(self):

        # perform sparse watershed assignment for component means
        watershed_modes = self.assign_modes(self.watershed_locs)
        watershed_assignments = torch.arange(
            self.hparams.num_components, device=watershed_modes.device
        )

        # loop over num_components to ensure all modes are correctly assigned
        # hierarchy of clusters cannot be longer than num_components
        for _ in range(self.hparams.num_components):
            watershed_assignments = watershed_modes[watershed_assignments]
        # reindex starting at 0
        unique_labels = torch.unique(watershed_assignments)
        for idx, label in enumerate(unique_labels):
            watershed_assignments[watershed_assignments == label] = idx

        return watershed_assignments

    def assign_labels(self, p):
        return self.watershed_assignments[self.assign_modes(p)]

    def on_train_start(self):
        with torch.no_grad():
            self.watershed_locs.copy_(self.locs)

    def on_train_end(self):
        self.watershed_assignments = self.watershed_labels()

    def configure_optimizers(self):
        optimizer = optim.Adam([self.watershed_locs], lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.hparams.scheduler_kwargs
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "entropy_epoch",
            }
        ]

    def on_train_epoch_end(self):
        self.log(
            "num_labels_epoch",
            self.watershed_labels().max() + 1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch, batch_idx):
        entropy = self.entropy_upper_bound()
        self.log(
            "entropy",
            entropy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return entropy

    def optimize_watershed(
        self,
        max_epochs=999,
        steps_per_epoch=10,
        patience=100,
        verbose=True,
        gpus=None,
        lr=1,
        scheduler_kwargs=None,
    ):
        self.hparams.lr = lr
        self.hparams.scheduler_kwargs = scheduler_kwargs or {}

        if verbose:
            print("optimizing entropy...")
        dummy_loader = DataLoader(np.zeros(steps_per_epoch), batch_size=1)
        early_stopping = pl.callbacks.EarlyStopping("entropy_epoch", patience=patience)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=verbose,
            enable_model_summary=False,
            callbacks=[early_stopping],
            accelerator="cpu" if gpus is None or gpus == 0 else "gpu",
            devices=1 if gpus is None or gpus == 0 else gpus,
            logger=False,
        )
        trainer.fit(self, dummy_loader)
