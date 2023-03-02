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

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr, mode
from sklearn.neighbors import KDTree
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm.autonotebook import tqdm

from selfsne.data import PairedDataset
from selfsne.nn import init_selu


class R2Score(nn.Module):
    def __init__(self, error):
        super().__init__()
        self.mean_ = None
        self.error = error

    def fit(self, x):
        self.mean_ = torch.tensor(x).mean(0, keepdims=True)

    def forward(self, input, target):
        input = torch.tensor(input)
        target = torch.tensor(target)
        res_error = self.error(input, target)
        total_error = self.error(self.mean_.expand(input.shape[0], -1), target)
        return torch.mean(1 - (res_error / total_error))


class SoftBCELoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        input = torch.stack([input, 1 - input], dim=1)
        target = torch.stack([target, 1 - target], dim=1)
        return super().forward(input, target)


def knn_probe_reconstruction(
    dataset,
    embedding,
    error,
    k=1,
    batch_size=1000,
    verbose=True,
    shuffle=False,
):
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)
    embedding_tree = KDTree(embedding)
    r2_score = R2Score(error)
    r2_score.fit(dataset)
    if verbose:
        prog_bar = tqdm(total=len(dataloader))
    y_pred = []
    for data, embedding_batch in dataloader:
        data = data.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.query(
            embedding_batch, k=k + 1, return_distance=False
        )[:, 1:]
        knn_data = dataset[knn_indices]
        knn = np.mean(knn_data, axis=1)
        y_pred.append(knn)
        if verbose:
            prog_bar.update(1)
    return r2_score(np.concatenate(y_pred), dataset).numpy()


def knn_probe_classification(
    labels,
    embedding,
    k=1,
    batch_size=1000,
    verbose=True,
    shuffle=False,
):
    paired_dataset = PairedDataset(labels, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)
    embedding_tree = KDTree(embedding)
    if verbose:
        prog_bar = tqdm(total=len(dataloader))
    y_pred = []
    for labels_batch, embedding_batch in dataloader:
        labels_batch = labels_batch.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.query(
            embedding_batch, k=k + 1, return_distance=False
        )[:, 1:]
        knn_labels = labels[knn_indices]
        knn = mode(knn_labels, axis=1)[0]
        y_pred.append(knn)
        if verbose:
            prog_bar.update(1)
    y_pred = np.concatenate(y_pred)
    f1 = f1_score(labels, y_pred, average="weighted")
    acc = balanced_accuracy_score(labels, y_pred, adjusted=True)
    return acc, f1


def linear_probe_reconstruction(
    dataset,
    embedding,
    loss,
    error,
    link=nn.Identity(),
    epochs=100,
    batch_size=1000,
    verbose=True,
    shuffle=False,
    tol=1e-3,
    lr=0.1,
):
    model = nn.Sequential(
        nn.BatchNorm1d(embedding.shape[1], momentum=None),
        init_selu(nn.Linear(embedding.shape[1], dataset.shape[1])),
        link,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
    )

    r2_score = R2Score(error)
    r2_score.fit(dataset)

    mean_loss_values = []
    if verbose:
        epoch_iter = tqdm(range(epochs), desc="Epoch")
        epoch_prog_bar = tqdm(
            total=len(dataloader), desc="Batch", leave=False, position=1
        )
    else:
        epoch_iter = range(epochs)

    patience = 0
    for epoch in epoch_iter:
        mean_loss = 0
        for data, embedding_batch in dataloader:
            data = data.float()
            embedding_batch = embedding_batch.float()
            optimizer.zero_grad()
            output = model(embedding_batch)
            batch_loss = loss(output, data)
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()
            mean_loss += batch_loss
            if verbose:
                epoch_prog_bar.update(1)
                epoch_prog_bar.set_postfix({"Loss": f"{batch_loss:.4f}"})
        if verbose:
            epoch_prog_bar.refresh()
            epoch_prog_bar.reset()
        mean_loss /= len(dataloader)
        mean_loss_values.append(mean_loss)
        if epoch > 1:
            if np.abs((mean_loss - mean_loss_values[-1]) / mean_loss_values[-1]) < tol:
                patience += 1
            else:
                patience = 0
            if patience > 2:
                break
    if verbose:
        epoch_prog_bar.close()

    metric_values = []
    model.eval()
    if verbose:
        prog_bar = tqdm(total=len(dataloader), desc="Eval")
    y_true = []
    y_pred = []
    for data, embedding_batch in dataloader:
        data = data.float()
        embedding_batch = embedding_batch.float()
        y_pred.append(model(embedding_batch).detach().numpy())
        y_true.append(data.numpy())
        if verbose:
            prog_bar.update(1)
    return r2_score(np.concatenate(y_pred), np.concatenate(y_true)).numpy()


def linear_probe_classification(
    labels,
    embedding,
    epochs=100,
    batch_size=1000,
    verbose=True,
    shuffle=False,
    tol=1e-3,
    lr=0.1,
):
    # Get number of classes
    num_classes = np.unique(labels).shape[0]
    loss = nn.CrossEntropyLoss()
    model = nn.Sequential(
        nn.BatchNorm1d(embedding.shape[1], momentum=None),
        init_selu(nn.Linear(embedding.shape[1], num_classes)),
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    paired_dataset = PairedDataset(labels, embedding, shuffle=shuffle)
    dataloader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
    )

    mean_loss_values = []

    if verbose:
        epoch_iter = tqdm(range(epochs), desc="Epoch")
        epoch_prog_bar = tqdm(
            total=len(dataloader), desc="Batch", leave=False, position=1
        )
    else:
        epoch_iter = range(epochs)

    patience = 0
    for epoch in epoch_iter:
        mean_loss = 0
        for label_batch, embedding_batch in dataloader:
            label_batch = label_batch.long()
            embedding_batch = embedding_batch.float()
            optimizer.zero_grad()
            output = model(embedding_batch)
            batch_loss = loss(output, label_batch)
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()
            mean_loss += batch_loss
            if verbose:
                epoch_prog_bar.update(1)
                epoch_prog_bar.set_postfix({"Loss": f"{batch_loss:.4f}"})
        if verbose:
            epoch_prog_bar.refresh()
            epoch_prog_bar.reset()

        epoch_prog_bar.reset()
        mean_loss /= len(dataloader)
        mean_loss_values.append(mean_loss)
        if epoch > 1:
            if np.abs((mean_loss - mean_loss_values[-1]) / mean_loss_values[-1]) < tol:
                patience += 1
            else:
                patience = 0
            if patience > 2:
                break
    if verbose:
        epoch_prog_bar.close()

    model.eval()
    if verbose:
        prog_bar = tqdm(total=len(dataloader), desc="Eval")

    y_true = []
    y_pred = []
    for label_batch, embedding_batch in dataloader:
        label_batch = label_batch.long()
        embedding_batch = embedding_batch.float()
        output = model(embedding_batch)
        _, predicted = torch.max(output, 1)
        y_true.extend(label_batch.tolist())
        y_pred.extend(predicted.tolist())
        if verbose:
            prog_bar.update(1)
    f1 = f1_score(y_true, y_pred, average="weighted")
    acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    return acc, f1


def pdist_no_diag(data):
    # compute pairwise distances
    distances = pdist(data)
    n = data.shape[0]
    # create a boolean mask of self-distances
    mask = (np.arange(len(distances)) % n) != ((np.arange(len(distances)) // n))
    # remove self-distance values
    distances = distances[mask]
    return distances


def pairwise_distance_correlation(
    dataset,
    embedding,
    batch_size=1000,
    num_batches=None,
    correlation="spearman",
    transform=lambda x: x,
    verbose=True,
    shuffle=False,
):
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)
    if num_batches is None or num_batches > len(dataloader):
        num_batches = len(dataloader)
    if verbose:
        prog_bar = tqdm(total=num_batches)
    data_distances = []
    embedding_distances = []
    for idx, (data, embedding_batch) in enumerate(dataloader):
        data = data.numpy()
        data_distances_ = transform(pdist_no_diag(data))
        embedding_distances_ = transform(pdist_no_diag(embedding_batch))
        data_distances.append(data_distances_)
        embedding_distances.append(embedding_distances_)
        if verbose:
            prog_bar.update(1)
        if idx >= num_batches - 1:
            if verbose:
                prog_bar.refresh()
                prog_bar.close()
            break
    if correlation == "spearman":
        rho, p = spearmanr(
            np.concatenate(data_distances), np.concatenate(embedding_distances)
        )
    elif correlation == "pearson":
        rho, p = pearsonr(
            np.concatenate(data_distances), np.concatenate(embedding_distances)
        )

    return rho


def knn_distance_correlation(
    dataset,
    embedding,
    k=1,
    batch_size=1000,
    num_batches=None,
    correlation="pearson",
    transform=lambda x: x,
    verbose=True,
    shuffle=False,
):
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)
    embedding_tree = KDTree(embedding)
    if num_batches is None or num_batches > len(dataloader):
        num_batches = len(dataloader)
    if verbose:
        prog_bar = tqdm(total=num_batches)
    data_distances = []
    embedding_distances = []
    for idx, (data, embedding_batch) in enumerate(dataloader):
        data = data.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.query(
            embedding_batch, k=k + 1, return_distance=False
        )[:, 1:]
        knn_embedding = embedding[knn_indices]
        knn_data = dataset[knn_indices]
        data_distances_ = transform(
            np.linalg.norm(data[:, np.newaxis] - knn_data, axis=-1)
        ).ravel()
        embedding_distances_ = transform(
            np.linalg.norm(embedding_batch[:, np.newaxis] - knn_embedding, axis=-1)
        ).ravel()
        data_distances.append(data_distances_)
        embedding_distances.append(embedding_distances_)
        if verbose:
            prog_bar.update(1)
        if idx >= num_batches - 1:
            if verbose:
                prog_bar.refresh()
                prog_bar.close()
            break
    if correlation == "spearman":
        rho, p = spearmanr(
            np.concatenate(data_distances), np.concatenate(embedding_distances)
        )
    if correlation == "pearson":
        rho, p = pearsonr(
            np.concatenate(data_distances), np.concatenate(embedding_distances)
        )
    return rho
