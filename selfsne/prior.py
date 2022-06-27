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

from selfsne.kernels import KERNELS
from selfsne.normalizers import NORMALIZERS
from selfsne.utils import disable_grad, enable_grad, stop_gradient


class MixturePrior(pl.LightningModule):
    def __init__(
        self,
        num_dims=2,
        num_components=2048,
        kernel="normal",
        logits="learn",
        kernel_scale=1.0,
        log_normalizer=0.0,
        lr=1.0,
        scheduler_kwargs={},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.kernel = KERNELS[kernel]

        if num_components == 1:
            locs = torch.zeros((num_components, num_dims))
            self.register_buffer("locs", locs)
        else:
            self.locs = nn.Parameter(torch.Tensor(num_components, num_dims))
            init.normal_(self.locs)

        if logits == "learn":
            self.logits = nn.Parameter(torch.zeros((num_components,)))
            init.zeros_(self.logits)
        elif logits == "maxent":
            logits = torch.zeros((num_components,))
            self.register_buffer("logits", logits)

        if isinstance(log_normalizer, str):
            self.log_normalizer = NORMALIZERS[log_normalizer]()
        else:
            self.log_normalizer = NORMALIZERS["constant"](log_normalizer)

        self.watershed_locs = nn.Parameter(stop_gradient(self.locs))
        self.watershed_assignments = self.watershed_labels()

    @property
    def multinomial(self):
        return D.Multinomial(logits=self.logits)

    @property
    def mixture(self):
        return D.Categorical(logits=self.logits)

    @property
    def components(self):
        return self.kernel(self.locs.unsqueeze(1), self.hparams.kernel_scale)

    def entropy(self):
        return -(self.mixture.probs * self._log_prob(self.locs)).sum()

    def weighted_log_prob(self, x):
        return self.components.log_prob(x) + self.mixture.logits.unsqueeze(1)

    def _log_prob(self, x):
        return self.weighted_log_prob(x).logsumexp(0)

    def log_prob(self, x):
        return self.log_normalizer(self._log_prob(x))

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
        self.watershed_locs.data = stop_gradient(self.locs.data)

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

    def training_epoch_end(self, training_step_outputs):
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
        scheduler_kwargs={},
    ):
        self.hparams.lr = lr
        self.hparams.scheduler_kwargs = scheduler_kwargs

        if verbose:
            print("optimizing entropy...")
        dummy_loader = DataLoader(np.zeros(steps_per_epoch), batch_size=1)
        early_stopping = pl.callbacks.EarlyStopping("entropy", patience=patience)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            progress_bar_refresh_rate=verbose,
            enable_model_summary=False,
            callbacks=[early_stopping],
            gpus=gpus,
        )
        trainer.fit(self, dummy_loader)
