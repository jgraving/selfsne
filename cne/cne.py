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
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
from torch.nn import Module, Linear
from torch.nn import init
from torch import optim

import numpy as np

from tqdm import tqdm
from collections import OrderedDict

from cne.likelihood import LIKELIHOODS
from cne.prior import MixturePrior
from cne.kernels import StudentTKernel, NormalKernel, VonMisesKernel
from cne.nn import FeedForward, init_linear
from cne.callbacks import EarlyStopping


def stop_gradient(x):
    return x.detach()


class Queue(nn.Module):
    def __init__(self, queue_size=4096):
        super(Queue, self).__init__()
        self.queue_size = queue_size
        self.queue = None

    def forward(self, x):
        if self.queue is None:
            self.queue = stop_gradient(x)
        else:
            self.queue = stop_gradient(torch.cat((self.queue, x))[-self.queue_size :])
        return self.queue

    def sample(self, n_samples):
        indices = torch.randint(
            self.queue.shape[0], (n_samples,), device=self.queue.device
        )
        return stop_gradient(self.queue[indices])

    @property
    def full(self):
        return self.queue.shape[0] == self.queue_size


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    # adapted from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CNE(Module):
    """
    Contrastive Noise Embedding (CNE).

    CNE uses self-supervised deep learning to create an embedding that
    maximizes the mutual information between each observation and samples
    from a distribution conditioned on the observation (i.e., observation + noise),
    such as sampling nearest neighbors or applying augmentations:

        x_i ~ p(x)
        y_i ~ p(y|x_i)

        I(x; y) ≥ InfoNCE(x; y) = E_p(x)p(y|x)[log q(y | x) − log q(y)]

    This implementation includes a learned Gaussian mixture prior for the embedding,
    which can be used for variational entropy clustering to automatically learn the
    number of clusters in the embedding. You can enable the fixed standard normal
    prior by setting `k_components` to 1.

    CNE also supports multiple similarity kernels to compute a variety of embedding types.

    Parameters
    ----------
    x_dim : int, default = None
        Dimensionality of the data. Default is None, which will
        infer dimensionality from the data when calling `.fit`
    z_dim : int, default = 2
        Dimensionality of the embedded space.
    k_components : int, default = 2048
        The number of mixture components for the prior distribution.
        Increasing this value increases the complexity of the prior
        distribution.
    metric : str, default = 'euclidean'
        The distance metric to use when calculating distance between
        data samples for calculating the high-dimensional similarities.
        It must be one one of ["euclidean" ...]
    rate_multiplier : float, default = 1
        The weighting for the rate loss function
    similarity_multiplier : float, default = 1
        The weighting for the contrastive loss
    likelihood_multiplier : float, default = 1
        The weighting for the distortion (negative log likelihood)
    likelihood : str, default = 'mse'
        The likelihood distribution for reconstructing the data.
        Must be one of likelihood.LIKELIHOODS
    kernel_type : str, default = "studentt"
        The kernel used for calculating low-dimensional pairwise similarities.
        Must be one of ["studentt", "normal", "vonmises"].
    encoder_layers : iterable, default = [256, 256, 256, 256]
        The layout of the hidden layers for the encoder. An iterable where each value
        corresponds to the number of units of one hidden layer in the network.
    decoder_layers : iterable, default = None
        The layout of the hidden layers for the decoder. An iterable where each value
        corresponds to the number of units of one hidden layer in the network.
        If `None` the decoder uses the encoder layout in reverse.
    device : str, default = "cpu"
        The torch-enabled device to use. To enable GPU support, set this to "cuda", or specify
        "cuda:0", "cuda:1", etc.
    prior_covariance : str, default = "eye"
        The method for calculating the covariance matrix for the mixture
        components of the prior. Only used when k_components > 1. Must be one of
        "eye", "iso", or "diag", which uses an identity matrix, learns a shared isotropic variance,
        or learns the diagonal as model parameters respectively.
    prior_mixture : str, default = "maxent"
        The method for calculating the weights for the mixture
        distribution of the prior. Only used when k_components > 1. Must be one of
        "maxent", or "learn", which uses a maximum entropy distribution, or
        learns the mixture weights as model parameters respectively.
    queue_size : int, default = 1000,
        The size of the online FIFO queue used for computing nearest neighbors.
        Larger values increase compute and memory requirements, but provide more accurate
        neighbor calculations.

    References
    ----------
    """

    def __init__(
        self,
        x_dim=None,
        z_dim=2,
        k_components=2048,
        encoder_layers=[256, 256, 256, 256],
        decoder_layers=None,
        kernel_type="studentt",
        metric="euclidean",
        likelihood="mse",
        device="cpu",
        similarity_multiplier=1.0,
        redundancy_multiplier=1.0,
        rate_multiplier=1.0,
        likelihood_multiplier=1.0,
        prior_mixture="maxent",
        prior_kernel="normal",
        queue_size=1000,
    ):
        super(CNE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.k_components = k_components
        self.encoder_layers = encoder_layers
        decoder_layers = (
            encoder_layers[::-1] if decoder_layers is None else decoder_layers
        )
        self.decoder_layers = decoder_layers
        self.kernel_type = kernel_type
        self.metric = metric
        self.likelihood = likelihood
        self.device = device
        self.similarity_multiplier = similarity_multiplier
        self.redundancy_multiplier = redundancy_multiplier
        self.rate_multiplier = rate_multiplier
        self.likelihood_multiplier = likelihood_multiplier
        self.initialized = False
        self.prior_mixture = prior_mixture
        self.prior_kernel = prior_kernel
        self.queue_size = queue_size

        if self.metric == "cross_entropy":
            self.distance = lambda x, y: -(
                x.log().softmax(-1) @ y.log().log_softmax(-1).T
            )
        elif self.metric == "inner_product":
            self.distance = lambda x, y: -(x @ y.T)
        elif self.metric == "euclidean":
            self.distance = torch.cdist
        elif self.metric == "mahalanobis":
            self.distance = lambda x, y: torch.cdist(self.input_bn(x), self.input_bn(y))
        elif self.metric == "correlation":
            self.distance = lambda x, y: -(self.input_bn(x) @ self.input_bn(y).T)

        if self.kernel_type == "normal":
            self.kernel = NormalKernel
        elif self.kernel_type == "studentt":
            self.kernel = StudentTKernel
        elif self.kernel_type == "categorical":
            self.kernel = CategoricalKernel
        elif self.kernel_type == "laplace":
            self.kernel = LaplaceKernel

    def init_model(self, in_features):

        if self.k_components > 1:
            self.prior = MixturePrior(
                self.z_dim,
                self.k_components,
                kernel=self.prior_kernel,
                logits_mode=self.prior_mixture,
            )
        else:
            self.prior = D.Independent(
                D.Normal(
                    torch.zeros((1, self.z_dim)).to(self.device),
                    torch.ones((1, self.z_dim)).to(self.device) * np.sqrt(self.z_dim),
                ),
                1,
            )

        self.encoder = FeedForward(
            in_features,
            self.encoder_layers,
        )
        self.embed = Linear(self.encoder_layers[-1], self.z_dim)
        init_linear(self.embed)

        self.input_bn = nn.BatchNorm1d(in_features, affine=False, momentum=None)
        self.queue = Queue(self.queue_size)

        self.decoder = FeedForward(
            self.z_dim,
            self.decoder_layers,
        )

        self.likelihood_fn = LIKELIHOODS[self.likelihood]()
        out_features = in_features * self.likelihood_fn.multiplier
        self.likelihood = Linear(self.decoder_layers[-1], out_features)
        init_linear(self.likelihood)

        self.bn = torch.nn.BatchNorm1d(self.z_dim, affine=False, momentum=None)

        self.to(self.device)
        self.initialized = True

    def distortion(self, x, z):
        decoded = self.decoder(self.bn(z))
        x_hat = self.likelihood(decoded)
        distortion = self.likelihood_fn(x, x_hat)
        return distortion

    def infonce(self, z_a, z_b):
        n = z_a.shape[0]
        kernel = self.kernel(z_a)
        similarity = -kernel.log_prob(loc=z_b, scale=np.sqrt(self.z_dim))
        contrast = kernel.log_prob(z_b.unsqueeze(-2))
        return similarity + (contrast.logsumexp(-1) - np.log(n))

    def redundancy_reduction(self, z_a, z_b):
        n = z_a.shape[0]
        d = z_a.shape[1]

        c_z = self.bn(z_a).T @ self.bn(z_b) / n
        invariance = (1 - c_z.diagonal()).pow(2).mean()  # .sum() / d
        redundancy = off_diagonal(c_z).pow(2).mean()  # .sum() / (d * d - d)

        return invariance + redundancy

    def forward(self, x):
        queue = self.queue(x)

        with torch.no_grad():
            values, indices = torch.topk(-self.distance(x, queue), 2, dim=-1)
            x_nn = queue[indices[:, -1]]
            # indices = D.Categorical(logits=-self.distance(x, queue)/self.temperature).sample()
            # x_nn = queue[indices]

        h_a = self.encoder(self.input_bn(x))
        z_a = self.embed(h_a)
        h_b = self.encoder(self.input_bn(x_nn))
        z_b = self.embed(h_b)

        similarity = self.infonce(z_a, z_b)
        redundancy = self.redundancy_reduction(z_a, z_b)
        distortion = self.distortion(x, z_a)
        rate = -self.prior.log_prob(z_a)

        return {
            "distortion": distortion.mean(),
            "rate": rate.mean(),
            "similarity": similarity.mean(),
            "redundancy": redundancy.mean(),
        }

    def train_batch(self, batch):
        batch = batch.to(self.device)
        batch_size, n_features = batch.shape
        loss = self(batch)

        loss["prior entropy"] = (
            -self.prior.entropy_lower_bound()
            if self.k_components > 1
            else torch.zeros_like(loss["similarity"])
        )

        loss["elbo"] = -(
            loss["distortion"] * self.likelihood_multiplier
            + loss["rate"] * self.rate_multiplier
        )

        loss["loss"] = (
            -loss["elbo"]
            + loss["similarity"] * self.similarity_multiplier
            + loss["redundancy"] * self.redundancy_multiplier
        )

        loss["unweighted_loss"] = (
            loss["distortion"] + loss["rate"] + loss["similarity"] + loss["redundancy"]
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
            "elbo": 0,
            "distortion": 0,
            "rate": 0,
            "similarity": 0,
            "redundancy": 0,
            "unweighted_loss": 0,
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
            "elbo",
            "distortion",
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

        if not self.initialized:
            self.init_model(x[0].shape[-1] if self.x_dim is None else self.x_dim)

        if restart_optimizer:
            params_list = [
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
                {"params": self.embed.parameters()},
                {"params": self.likelihood.parameters()},
                {"params": self.input_bn.parameters()},
                {"params": self.bn.parameters()},
            ]
            if self.k_components > 1:
                params_list.append(
                    {"params": self.prior.parameters(), "weight_decay": 0.0, "lr": lr}
                )

            self.optimizer = optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)

        if self.k_components > 1:
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
            if self.queue.full:
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
        log_likelihood = []
        prior_log_prob = []
        if self.k_components > 1:
            labels = []
        predict_loader = torch.utils.data.DataLoader(
            x, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if self.k_components > 1 and return_labels:
            if not watershed_kwargs:
                watershed_kwargs = {"lr": 0.01, "patience": 10, "n_iter": 9999}
            if not self.prior.watershed_optimized:
                self.prior.optimize_watershed(verbose=verbose, **watershed_kwargs)
        if verbose:
            predict_loader = tqdm(predict_loader, desc="prediction")

        self.eval()
        for batch in predict_loader:
            batch = batch.to(self.device)
            embedded = self.embed(self.encoder(self.input_bn(batch)))
            distortion = self.distortion(batch, embedded)
            prior_log_prob.append(self.prior.log_prob(embedded).cpu())
            if self.k_components > 1 and return_labels:
                classes = self.prior.assign_labels(embedded)
                labels.append(classes.cpu())

            embedding.append(embedded.cpu())
            log_likelihood.append(distortion.cpu())

        return {
            "embedding": np.concatenate(embedding),
            "log_likelihood": np.concatenate(log_likelihood),
            "labels": np.concatenate(labels)
            if self.k_components > 1 and return_labels
            else None,
            "prior_log_prob": np.concatenate(prior_log_prob),
        }
