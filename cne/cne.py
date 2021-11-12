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
from torch import nn
from torch.optim import AdamW

import numpy as np

from tqdm import tqdm
from collections import OrderedDict

from cne.prior import MixturePrior
from cne.kernels import KERNELS
from cne.callbacks import EarlyStopping
from cne.losses import infonce, redundancy_reduction
from cne.neighbors import NearestNeighborSampler


class CNE(nn.Module):
    """
    Contrastive Noise Embedding (CNE).

    CNE uses self-supervised deep learning to create an embedding that
    maximizes the mutual information between each observation and samples
    from a distribution conditioned on the observation (i.e., observation + noise),
    such as sampling nearest neighbors or applying augmentations:

        x_i ~ p(x)
        y_i ~ p(y|x_i)

        I(x; y) ≥ InfoNCE(x; y) = p(x)p(y|x) log q(y | x) − log q(y)

    Parameters
    ----------
    encoder : torch.nn.Module
        An encoder module that transforms data from x_dim to z_dim
    x_dim : int, default = None
        Dimensionality of the data. Default is None, which will
        infer dimensionality from the data when calling `.fit`
    z_dim : int, default = 2
        Dimensionality of the embedded space.
    rate_multiplier : float, default = 1
        The weighting for the rate loss function
    similarity_multiplier : float, default = 1
        The weighting for the contrastive loss
    redundancy_multiplier : float, default = 1
        The weighting for the orthogonal loss
    kernel : str, default = "studentt"
        The kernel used for calculating low-dimensional pairwise similarities.
        Must be one of cne.kernels.KERNELS.
    device : str, default = "cpu"
        The torch-enabled device to use. To enable GPU support, set this to "cuda", or specify
        "cuda:0", "cuda:1", etc.

    References
    ----------
    """

    def __init__(
        self,
        encoder,
        x_dim=None,
        z_dim=2,
        kernel="studentt",
        prior=MixturePrior(),
        device="cpu",
        similarity_multiplier=1.0,
        redundancy_multiplier=1.0,
        rate_multiplier=1.0,
        noise=NearestNeighborSampler(),
    ):
        super(CNE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.similarity_multiplier = similarity_multiplier
        self.redundancy_multiplier = redundancy_multiplier
        self.rate_multiplier = rate_multiplier
        self.noise = noise
        self.encoder = encoder
        self.kernel = KERNELS[kernel]
        self.prior = prior
        self.bn = nn.BatchNorm1d(z_dim, affine=False)
        self.to(self.device)

    def forward(self, x_a):
        x_b = self.noise(x_a)
        z_a = self.encoder(x_a)
        z_b = self.encoder(x_b)

        similarity = infonce(z_a, z_b, self.kernel)
        redundancy = redundancy_reduction(z_a, z_b, self.bn)
        rate = -self.prior.log_prob(z_a)

        return {
            "rate": rate.mean(),
            "similarity": similarity.mean(),
            "redundancy": redundancy.mean(),
        }

    def train_batch(self, batch):
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        loss = self(batch)

        loss["prior entropy"] = self.prior.entropy()

        loss["loss"] = (
            loss["rate"] * self.rate_multiplier
            + loss["similarity"] * self.similarity_multiplier
            + loss["redundancy"] * self.redundancy_multiplier
        )

        for metric in loss.keys():
            loss[metric] = loss[metric].mean()

        self.optimizer.zero_grad()
        loss["loss"].backward()
        self.optimizer.step()

        return loss

    def train_epoch(
        self, epoch_idx, train_loader, progress_bar, early_stopping, verbose
    ):

        epoch_metrics = {
            "loss": 0,
            "rate": 0,
            "similarity": 0,
            "redundancy": 0,
            "prior entropy": 0,
        }

        if verbose:
            n_epochs = len(progress_bar)
            description = "Epoch: {}/{}  Batch: {}/{}"
            progress_bar.set_description(
                description.format(epoch_idx, n_epochs, 1, len(train_loader))
            )

        for batch_idx, batch in enumerate(train_loader):
            batch_metrics = self.train_batch(batch)
            if verbose:
                for metric in epoch_metrics.keys():
                    epoch_metrics[metric] += batch_metrics[metric].mean().item()

                display_metrics = {}
                for metric in epoch_metrics.keys():
                    display_metrics[metric] = epoch_metrics[metric] / (batch_idx + 1)

                progress_bar.set_description(
                    description.format(
                        epoch_idx, n_epochs, batch_idx + 1, len(train_loader)
                    )
                )
                self.progress_bar_step(progress_bar, display_metrics, early_stopping)

        for metric in epoch_metrics.keys():
            epoch_metrics[metric] /= batch_idx + 1

        return epoch_metrics

    def progress_bar_step(self, progress_bar, metrics, early_stopping):
        display_metrics = OrderedDict({})
        metric_names = [
            "loss",
            "rate",
            "similarity",
            "redundancy",
            "prior entropy",
        ]
        for metric in metric_names:
            display_metrics[metric] = "{:.3f}".format(metrics[metric])
        display_metrics["no improvement"] = "{}/{}".format(
            early_stopping.num_bad_epochs, early_stopping.patience
        )
        progress_bar.set_postfix(display_metrics)

    def fit(
        self,
        x,
        n_epochs=1,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-2,
        verbose=1,
        monitor="similarity",
        threshold=1e-3,
        patience=10,
        threshold_mode="rel",
        num_workers=0,
        drop_last=True,
        restart_optimizer=True,
    ):
        """
        Trains the model for a fixed number of epochs, or
        until early stopping is initialized.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        batch_size : int, default = 512
            Number of samples per gradient update.
        epochs : int, default = 1
            The number of times to iterate over the training data
        lr : float, default = 1e-3
            The learning rate for the Adam optimizer
        weight_decay : float, default = 1e-2
            The l2 weight decay for regularizing the model parameters
        verbose : int, default = 1
            Verbosity mode. 0 = silent, ≥1 = verbose
        monitor : str, default = "similarity"
            The metric to monitor for early stopping.
        threshold : float, default = 1e-3
            The minimum improvement needed for prevent early stopping.
        threshold_mode : str, default = 'rel'
            One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 - threshold ).
            In `abs` mode, dynamic_threshold = best - threshold in `min` mode.
        patience : int, default = 10,
            Number of epochs with no improvement after
            which training will be stopped. For example, if patience = 2,
            then it will ignore the first 2 epochs with no improvement,
            and will only stop training after the 3rd epoch if the loss
            still hasn't improved then. Setting patience = 0 turns off early stopping.
            Increase this if the dataset is small.
        num_workers : int, default = 0
            How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
        """

        if restart_optimizer:
            params_list = [
                {"params": self.encoder.parameters()},
                {"params": self.bn.parameters()},
            ]
            params_list.append({"params": self.prior.parameters(), "weight_decay": 0.0})

            self.optimizer = AdamW(params_list, lr=lr, weight_decay=weight_decay)

        self.prior.watershed_optimized = False

        train_loader = torch.utils.data.DataLoader(
            x,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
        )

        self.train()
        epochs = range(1, n_epochs + 1)
        if verbose:
            epochs = tqdm(epochs)
        early_stopping = EarlyStopping(
            threshold=threshold, patience=patience, threshold_mode=threshold_mode
        )
        for epoch_idx in epochs:
            epoch_metrics = self.train_epoch(
                epoch_idx,
                train_loader,
                progress_bar=epochs,
                early_stopping=early_stopping,
                verbose=verbose,
            )
            if early_stopping.step(epoch_metrics[monitor]):
                break

    @torch.no_grad()
    def predict(
        self,
        x,
        batch_size=1024,
        verbose=1,
        num_workers=0,
        return_labels=True,
        watershed_kwargs=None,
    ):
        """
        Predicts on new data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and
            n_features is the number of features.
        batch_size : int, default = 1024
            Number of samples to compute at once.
        verbose : int, default = 0
            Verbosity mode. 0 = silent, ≥1 = verbose
        num_workers : int, default = 0
            How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.

        Returns
        -------
        results : dict
            Returns dict containing model outputs, :
                "mean": the mean of the approx. posterior for each sample
                "variance": the variance of the approx. posterior for each sample
                "log_likelihood": the log likelihood score for each sample
                "entropy": the entropy of the approx. posterior for each sample
        """

        embedding = []
        prior_log_prob = []
        predict_loader = torch.utils.data.DataLoader(
            x, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if return_labels:
            labels = []
            if not watershed_kwargs:
                watershed_kwargs = {"lr": 0.01, "patience": 10, "n_iter": 9999}
            if not self.prior.watershed_optimized:
                self.prior.optimize_watershed(verbose=verbose, **watershed_kwargs)

        if verbose:
            predict_loader = tqdm(predict_loader, desc="prediction")

        self.eval()
        for batch in predict_loader:
            batch = batch.to(self.device)
            embedded = self.encoder(batch)
            prior_log_prob.append(self.prior.log_prob(embedded).cpu())
            if return_labels:
                classes = self.prior.assign_labels(embedded)
                labels.append(classes.cpu())

            embedding.append(embedded.cpu())

        return {
            "embedding": np.concatenate(embedding),
            "labels": np.concatenate(labels) if return_labels else None,
            "prior_log_prob": np.concatenate(prior_log_prob),
        }
