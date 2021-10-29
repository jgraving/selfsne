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
import torch.nn.functional as F
import torch.distributions as D
from torch.nn import Module, Parameter
from torch.nn import init
from torch import optim

import numpy as np
from collections import OrderedDict

from cne.callbacks import EarlyStopping


class MixturePrior(Module):
    def __init__(
        self,
        z_dim,
        k_components,
        kernel="normal",
        logits_mode="maxent",
    ):
        super(MixturePrior, self).__init__()
        self.z_dim = z_dim
        self.k_components = k_components
        if kernel == "studentt":
            self.kernel = StudentTKernel
        elif kernel == "normal":
            self.kernel = NormalKernel

        self.locs = Parameter(torch.Tensor(k_components, z_dim))
        init.normal_(self.locs, std=np.sqrt(self.z_dim))

        if logits_mode == "learn":
            self.logits = Parameter(torch.zeros((k_components,)))
            init.zeros_(self.logits)
        elif logits_mode == "maxent":
            logits = torch.zeros((k_components,))
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
        return self.kernel(loc=self.locs, scale=np.sqrt(self.z_dim))

    def entropy_lower_bound(self):
        return (self.mixture.probs * self.log_prob(self.locs)).sum()

    def weighted_log_prob(self, x):
        return self.components.log_prob(x.unsqueeze(-2)) + self.mixture.logits

    def log_prob(self, x):
        return self.weighted_log_prob(x).logsumexp(-1)

    def assign_modes(self, x):
        return self.weighted_log_prob(x).argmax(-1)

    def soft_quantize(self, x):
        indices = D.Categorical(logits=self.weighted_log_prob(x)).sample()
        return self.locs[indices]

    def hard_quantize(self, x):
        return self.locs[self.assign_modes(x)]

    def gradient_quantize(self, x):
        return (self.soft_quantize(x) - x).detach() + x

    @torch.enable_grad()
    def optimize_watershed(
        self, verbose=True, lr=0.1, patience=10, n_iter=9999, steps_per_epoch=100
    ):
        self.watershed_locs = Parameter(self.locs.clone())
        params = [self.watershed_locs]

        optimizer = optim.Adam(params, lr=lr)
        early_stopping = EarlyStopping(
            patience=patience, threshold_mode="abs", threshold=1e-3
        )
        self.train()
        if verbose:
            progress_bar = tqdm(range(n_iter), desc="entropy optimization")
        else:
            progess_bar = range(n_iter)

        display_metrics = OrderedDict({})

        for idx in progress_bar:
            total_loss = 0
            for jdx in range(steps_per_epoch):
                loss = -(self.mixture.probs * self.log_prob(self.watershed_locs)).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (jdx + 1)
                if verbose:
                    display_metrics["entropy"] = "{:.3f}".format(avg_loss)
                    display_metrics["no improvement"] = "{}/{}".format(
                        early_stopping.num_bad_epochs, early_stopping.patience
                    )
                    progress_bar.set_postfix(display_metrics)
            if early_stopping.step(avg_loss):
                break

        # calculate watershed basins for each component
        watershed_modes = self.assign_modes(self.watershed_locs)
        # perform sparse watershed assignment for component means
        watershed_assignments = torch.arange(self.k_components).to(
            watershed_modes.device
        )
        # loop over k_components to ensure all modes are correctly assigned
        # hierarchy of clusters cannot be longer than k_components
        for _ in range(self.k_components):
            watershed_assignments = watershed_modes[watershed_assignments]
        # reindex starting at 0
        unique_labels = torch.unique(watershed_assignments)
        for idx, label in enumerate(unique_labels):
            watershed_assignments[watershed_assignments == label] = idx
        self.watershed_assignments = watershed_assignments
        self.watershed_optimized = True

    def assign_labels(self, p):
        if not self.watershed_optimized:
            self.optimize_watershed()
        return self.watershed_assignments[self.assign_modes(p)]
