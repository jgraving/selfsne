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
from torch.nn import init
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

from cne.kernels import KERNELS


class MixturePrior(pl.LightningModule):
    def __init__(self, z_dim=2, k_components=2048, kernel="normal", logits="learn"):
        super().__init__()
        self.save_hyperparameters()
        self.kernel = KERNELS[self.hparams.kernel]

        if self.hparams.k_components == 1:
            locs = torch.zeros((self.hparams.k_components, self.hparams.z_dim))
            self.register_buffer("locs", locs)
        else:
            self.locs = nn.Parameter(
                torch.Tensor(self.hparams.k_components, self.hparams.z_dim)
            )
            init.normal_(self.locs, std=1 / np.sqrt(self.hparams.k_components))

        if self.hparams.logits == "learn":
            self.logits = nn.Parameter(torch.zeros((self.hparams.k_components,)))
            init.zeros_(self.logits)
        elif self.hparams.logits == "maxent":
            logits = torch.zeros((self.hparams.k_components,))
            self.register_buffer("logits", logits)

        self.watershed_optimized = False

    @property
    def multinomial(self):
        return D.Multinomial(logits=self.logits)

    @property
    def mixture(self):
        return D.Categorical(logits=self.logits)

    @property
    def components(self):
        return self.kernel(self.locs)

    def entropy(self):
        return -(self.mixture.probs * self.log_prob(self.locs)).sum()

    def weighted_log_prob(self, x):
        return self.components.log_prob(x.unsqueeze(-2)) + self.mixture.logits

    def log_prob(self, x):
        return self.weighted_log_prob(x).logsumexp(-1)

    def assign_modes(self, x):
        return self.weighted_log_prob(x).argmax(-1)

    def configure_optimizers(self):
        self.watershed_locs = nn.Parameter(self.locs.clone())
        return optim.Adam([self.watershed_locs], lr=self.hparams.lr)

    def entropy_upper_bound(self):
        return -(self.mixture.probs * self.log_prob(self.watershed_locs)).sum()

    def watershed_labels(self):
        watershed_modes = self.assign_modes(self.watershed_locs)
        # perform sparse watershed assignment for component means
        watershed_assignments = torch.arange(self.hparams.k_components).to(
            watershed_modes.device
        )
        # loop over k_components to ensure all modes are correctly assigned
        # hierarchy of clusters cannot be longer than k_components
        for _ in range(self.hparams.k_components):
            watershed_assignments = watershed_modes[watershed_assignments]
        # reindex starting at 0
        unique_labels = torch.unique(watershed_assignments)
        for idx, label in enumerate(unique_labels):
            watershed_assignments[watershed_assignments == label] = idx

        return watershed_assignments

    def on_train_end(self):
        self.watershed_assignments = self.watershed_labels()
        self.watershed_optimized = True

    def training_step(self, batch, batch_idx):
        self.log(
            "n_labels",
            self.watershed_labels().max() + 1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "entropy",
            self.entropy_upper_bound(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return self.entropy_upper_bound()

    def assign_labels(self, p):
        if not self.watershed_optimized:
            self.optimize_watershed()
        return self.watershed_assignments[self.assign_modes(p)]

    def optimize_watershed(
        self,
        max_epochs=999,
        steps_per_epoch=10,
        patience=10,
        verbose=True,
        gpus=None,
        lr=0.1,
    ):
        self.hparams.lr = lr
        if verbose:
            print("optimizing entropy...")
        dummy_loader = DataLoader(np.zeros(steps_per_epoch), batch_size=1)
        early_stopping = pl.callbacks.EarlyStopping("entropy", patience=patience)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            progress_bar_refresh_rate=verbose,
            weights_summary=None,
            callbacks=[early_stopping],
            gpus=gpus,
        )
        trainer.fit(self, dummy_loader)
