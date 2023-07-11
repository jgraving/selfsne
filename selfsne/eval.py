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
import numba
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr, mode
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm
from copy import deepcopy
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


def knn_reconstruction(
    train_dataset,
    train_embedding,
    test_dataset,
    test_embedding,
    error,
    k=1,
    batch_size=1000,
    verbose=True,
):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    embedding_tree = NearestNeighbors(n_jobs=-1)
    embedding_tree.fit(train_embedding)
    r2_score = R2Score(error)
    r2_score.fit(train_dataset)
    if verbose:
        train_prog_bar = tqdm(total=len(train_dataloader), desc="Train")
        test_prog_bar = tqdm(total=len(test_dataloader), desc="Test")

    train_pred, test_pred = [], []

    # for train data
    for i, train_data in enumerate(train_dataloader):
        train_data = train_data.numpy()
        train_embedding_batch = train_embedding[i * batch_size : (i + 1) * batch_size]
        train_knn_indices = embedding_tree.kneighbors(
            train_embedding_batch, n_neighbors=k + 1, return_distance=False
        )[:, 1:]
        train_knn_data = train_dataset[train_knn_indices]
        train_knn = np.mean(train_knn_data, axis=1)
        train_pred.append(train_knn)
        if verbose:
            train_prog_bar.update(1)

    # for test data
    for i, test_data in enumerate(test_dataloader):
        test_data = test_data.numpy()
        test_embedding_batch = test_embedding[i * batch_size : (i + 1) * batch_size]
        test_knn_indices = embedding_tree.kneighbors(
            test_embedding_batch, n_neighbors=k + 1, return_distance=False
        )[:, 1:]
        test_knn_data = train_dataset[
            test_knn_indices
        ]  # using the train_dataset to retrieve the neighbors
        test_knn = np.mean(test_knn_data, axis=1)
        test_pred.append(test_knn)
        if verbose:
            test_prog_bar.update(1)

    train_score = r2_score(np.concatenate(train_pred), train_dataset).item()
    test_score = r2_score(np.concatenate(test_pred), test_dataset).item()

    return train_score, test_score


def knn_classification(
    train_labels,
    train_embedding,
    test_labels,
    test_embedding,
    k=1,
    batch_size=1000,
    verbose=True,
    shuffle=False,
):
    train_dataloader = DataLoader(train_labels, batch_size=batch_size)
    test_dataloader = DataLoader(test_labels, batch_size=batch_size)
    embedding_tree = NearestNeighbors(n_jobs=-1)
    embedding_tree.fit(train_embedding)
    if verbose:
        train_prog_bar = tqdm(total=len(train_dataloader), desc="Train")
        test_prog_bar = tqdm(total=len(test_dataloader), desc="Test")

    train_pred, test_pred = [], []
    # for train data
    for labels_batch, embedding_batch in train_dataloader:
        labels_batch = labels_batch.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.kneighbors(
            embedding_batch, n_neighbors=k + 1, return_distance=False
        )[:, 1:]
        knn_labels = train_labels[knn_indices]
        train_pred.append(mode(knn_labels, axis=1)[0])
        if verbose:
            train_prog_bar.update(1)

    # for test data
    for labels_batch, embedding_batch in test_dataloader:
        labels_batch = labels_batch.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.kneighbors(
            embedding_batch, n_neighbors=k + 1, return_distance=False
        )
        knn_labels = train_labels[knn_indices]
        test_pred.append(mode(knn_labels, axis=1)[0])
        if verbose:
            test_prog_bar.update(1)

    train_acc = balanced_accuracy_score(
        train_labels, np.concatenate(train_pred), adjusted=True
    )
    train_f1 = f1_score(train_labels, np.concatenate(train_pred), average="weighted")
    train_precision = precision_score(
        train_labels, np.concatenate(train_pred), average="weighted"
    )
    train_recall = recall_score(
        train_labels, np.concatenate(train_pred), average="weighted"
    )

    test_acc = balanced_accuracy_score(
        test_labels, np.concatenate(test_pred), adjusted=True
    )
    test_f1 = f1_score(test_labels, np.concatenate(test_pred), average="weighted")
    test_precision = precision_score(
        test_labels, np.concatenate(test_pred), average="weighted"
    )
    test_recall = recall_score(
        test_labels, np.concatenate(test_pred), average="weighted"
    )

    return (train_acc, train_f1, train_precision, train_recall), (
        test_acc,
        test_f1,
        test_precision,
        test_recall,
    )


def linear_reconstruction(
    train_dataset,
    train_embedding,
    test_dataset,
    test_embedding,
    loss,
    error,
    model=None,
    link=nn.Identity(),
    epochs=100,
    batch_size=1024,
    verbose=True,
    shuffle=False,
    tol=1e-2,
    lr=0.3,
    patience=1,
):
    if model is None:
        model = init_selu(nn.Linear(train_embedding.shape[1], train_dataset.shape[1]))
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scaler = StandardScaler().fit(train_embedding)
    normalized_train_embedding = scaler.transform(train_embedding)
    paired_train_dataset = PairedDataset(
        train_dataset, normalized_train_embedding, shuffle=shuffle
    )

    normalized_test_embedding = scaler.transform(test_embedding)
    paired_test_dataset = PairedDataset(
        test_dataset, normalized_test_embedding, shuffle=shuffle
    )

    train_dataloader = DataLoader(
        paired_train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
    )
    eval_train_dataloader = DataLoader(
        paired_train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )
    eval_test_dataloader = DataLoader(
        paired_test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )

    r2_score = R2Score(error)
    r2_score.fit(train_dataset)

    mean_loss_values = []
    if verbose:
        epoch_iter = tqdm(range(epochs), desc="Epoch")
        epoch_prog_bar = tqdm(
            total=len(train_dataloader), desc="Batch", leave=False, position=1
        )
    else:
        epoch_iter = range(epochs)

    no_improvement = 0
    best_loss = np.inf
    for epoch in epoch_iter:
        mean_loss = 0
        for data, embedding_batch in train_dataloader:
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
                epoch_prog_bar.set_postfix(
                    {
                        "Loss": f"{batch_loss:.4f}",
                        "Best": f"{best_loss:.4f}",
                        "Patience": f"{no_improvement}/{patience}",
                    }
                )
        if verbose:
            epoch_prog_bar.refresh()
            epoch_prog_bar.reset()
        mean_loss /= len(train_dataloader)
        mean_loss_values.append(mean_loss)
        if epoch >= 1:
            rel_improvement = np.abs(mean_loss - best_loss / best_loss)
            if rel_improvement < tol or (
                (mean_loss > best_loss) and (rel_improvement > tol)
            ):
                no_improvement += 1
            else:
                no_improvement = 0
            if no_improvement > patience:
                break
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_model = deepcopy(model)
    if verbose:
        epoch_prog_bar.close()

    metric_values = []
    model.eval()

    # Evaluation on train data
    y_true = []
    y_pred = []
    for data, embedding_batch in eval_train_dataloader:
        data = data.float()
        embedding_batch = embedding_batch.float()
        y_pred.append(link(best_model(embedding_batch)).detach().numpy())
        y_true.append(data.numpy())
    train_r2_score = r2_score(np.concatenate(y_pred), np.concatenate(y_true)).item()

    # Evaluation on test data
    y_true = []
    y_pred = []
    for data, embedding_batch in eval_test_dataloader:
        data = data.float()
        embedding_batch = embedding_batch.float()
        y_pred.append(link(best_model(embedding_batch)).detach().numpy())
        y_true.append(data.numpy())
    test_r2_score = r2_score(np.concatenate(y_pred), np.concatenate(y_true)).item()

    return train_r2_score, test_r2_score


def linear_classification(
    train_labels,
    train_embedding,
    test_labels,
    test_embedding,
    epochs=100,
    batch_size=1024,
    verbose=True,
    shuffle=False,
    tol=1e-2,
    lr=0.3,
    patience=1,
):
    # Get number of classes
    num_classes = np.unique(train_labels).shape[0]

    loss = nn.CrossEntropyLoss()
    model = init_selu(nn.Linear(train_embedding.shape[1], num_classes))
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    normalized_train_embedding = StandardScaler().fit_transform(train_embedding)
    paired_train_dataset = PairedDataset(
        train_labels, normalized_train_embedding, shuffle=shuffle
    )

    normalized_test_embedding = StandardScaler().fit_transform(test_embedding)
    paired_test_dataset = PairedDataset(
        test_labels, normalized_test_embedding, shuffle=shuffle
    )

    train_dataloader = DataLoader(
        paired_train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
    )
    eval_train_dataloader = DataLoader(
        paired_train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )
    eval_test_dataloader = DataLoader(
        paired_test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )

    mean_loss_values = []

    if verbose:
        epoch_iter = tqdm(range(epochs), desc="Epoch")
        epoch_prog_bar = tqdm(
            total=len(train_dataloader), desc="Batch", leave=False, position=1
        )
    else:
        epoch_iter = range(epochs)

    no_improvement = 0
    best_loss = np.inf
    for epoch in epoch_iter:
        mean_loss = 0
        for label_batch, embedding_batch in train_dataloader:
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
                epoch_prog_bar.set_postfix(
                    {
                        "Loss": f"{batch_loss:.4f}",
                        "Best": f"{best_loss:.4f}",
                        "Patience": f"{no_improvement}/{patience}",
                    }
                )
        if verbose:
            epoch_prog_bar.refresh()
            epoch_prog_bar.reset()

        mean_loss /= len(train_dataloader)
        mean_loss_values.append(mean_loss)
        if epoch >= 1:
            rel_improvement = np.abs(mean_loss - best_loss / best_loss)
            if rel_improvement < tol or (
                (mean_loss > best_loss) and (rel_improvement > tol)
            ):
                no_improvement += 1
            else:
                no_improvement = 0
            if no_improvement > patience:
                break
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_model = deepcopy(model)

    if verbose:
        epoch_prog_bar.close()
    model.eval()

    # Evaluation on train data
    y_true = []
    y_pred = []
    for label_batch, embedding_batch in eval_train_dataloader:
        label_batch = label_batch.long()
        embedding_batch = embedding_batch.float()
        output = best_model(embedding_batch)
        _, predicted = torch.max(output, 1)
        y_true.extend(label_batch.tolist())
        y_pred.extend(predicted.tolist())
    f1_train = f1_score(y_true, y_pred, average="weighted")
    acc_train = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    precision_train = precision_score(y_true, y_pred, average="weighted")
    recall_train = recall_score(y_true, y_pred, average="weighted")

    # Evaluation on test data
    y_true = []
    y_pred = []
    for label_batch, embedding_batch in eval_test_dataloader:
        label_batch = label_batch.long()
        embedding_batch = embedding_batch.float()
        output = best_model(embedding_batch)
        _, predicted = torch.max(output, 1)
        y_true.extend(label_batch.tolist())
        y_pred.extend(predicted.tolist())
    f1_test = f1_score(y_true, y_pred, average="weighted")
    acc_test = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    precision_test = precision_score(y_true, y_pred, average="weighted")
    recall_test = recall_score(y_true, y_pred, average="weighted")

    return (
        acc_train,
        f1_train,
        precision_train,
        recall_train,
        acc_test,
        f1_test,
        precision_test,
        recall_test,
    )


def pdist_no_diag(data):
    # compute pairwise distances
    distances = pdist(data)
    n = data.shape[0]
    # create a boolean mask of self-distances
    mask = (np.arange(len(distances)) % n) != ((np.arange(len(distances)) // n))
    # remove self-distance values
    return distances[mask]


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
    embedding_tree = NearestNeighbors(n_jobs=-1)
    embedding_tree.fit(embedding)
    if num_batches is None or num_batches > len(dataloader):
        num_batches = len(dataloader)
    if verbose:
        prog_bar = tqdm(total=num_batches)
    data_distances = []
    embedding_distances = []
    for idx, (data, embedding_batch) in enumerate(dataloader):
        data = data.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.kneighbors(
            embedding_batch, n_neighbors=k + 1, return_distance=False
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


@numba.njit(fastmath=True)
def sum_isin(x, y):
    result = 0
    for idx in range(x.shape[0]):
        if x[idx] in y:
            result += 1
    return result


@numba.njit(parallel=True, fastmath=True)
def set_intersection(x, y):
    intersection = 0
    for idx in numba.prange(x.shape[0]):
        intersection += sum_isin(x[idx], y[idx])
    return intersection


def neighborhood_preservation(
    dataset,
    embedding,
    k=1,
    batch_size=1000,
    num_batches=None,
    verbose=True,
    shuffle=False,
):
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)
    embedding_tree = NearestNeighbors(n_jobs=-1)
    embedding_tree.fit(embedding)
    dataset_tree = NearestNeighbors(n_jobs=-1)
    dataset_tree.fit(dataset)
    if num_batches is None or num_batches > len(dataloader):
        num_batches = len(dataloader)
    if verbose:
        prog_bar = tqdm(total=num_batches)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for idx, (data, embedding_batch) in enumerate(dataloader):
        data = data.numpy()
        embedding_batch = embedding_batch.numpy()
        embedding_knn_indices = embedding_tree.kneighbors(
            embedding_batch, n_neighbors=k + 1, return_distance=False
        )[:, 1:]
        dataset_knn_indices = dataset_tree.kneighbors(
            data, n_neighbors=k + 1, return_distance=False
        )[:, 1:]

        # Perform set intersection for each row in the indices
        intersection = set_intersection(embedding_knn_indices, dataset_knn_indices)
        tp = intersection
        fp = embedding_knn_indices.size - intersection
        fn = dataset_knn_indices.size - intersection
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        if verbose:
            prog_bar.update(1)
        if idx >= num_batches - 1:
            if verbose:
                prog_bar.refresh()
                prog_bar.close()
            break
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall
