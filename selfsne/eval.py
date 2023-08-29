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

from sklearn.neighbors import (
    NearestNeighbors,
    KNeighborsRegressor,
    KNeighborsClassifier,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from tqdm.autonotebook import tqdm

from copy import deepcopy

from selfsne.data import PairedDataset
from selfsne.nn import init_selu


class R2Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.error = nn.MSELoss()

    def fit(self, x):
        self.mean_ = torch.tensor(x).mean(0, keepdims=True)

    def forward(self, input, target):
        input = torch.tensor(input)
        target = torch.tensor(target)
        res_error = self.error(input, target)
        total_error = self.error(self.mean_.expand(input.shape[0], -1), target)
        return torch.mean(1 - (res_error / total_error))


def knn_reconstruction(
    train_dataset, train_embedding, test_dataset, test_embedding, k=1, verbose=True, metric='minkowski'
):
    if verbose:
        print("Fitting k-NN regressor...")

    # Initialize the k-NN regressor for test set with n_neighbors=k
    knn_regressor = KNeighborsRegressor(n_neighbors=k, n_jobs=-1, metric=metric)
    knn_regressor.fit(train_embedding, train_dataset)

    r2_score = R2Score()
    r2_score.fit(train_dataset)

    if verbose:
        print("Making predictions on testing data...")

    # Make predictions on the testing data
    test_pred = knn_regressor.predict(test_embedding)

    # Re-fit the k-NN regressor for training set with n_neighbors=k+1
    knn_regressor.set_params(n_neighbors=k + 1)
    knn_regressor.fit(train_embedding, train_dataset)

    if verbose:
        print("Making predictions on training data...")

    # Compute the (k + 1) nearest neighbors for the training data
    train_neighbors = knn_regressor.kneighbors(train_embedding, return_distance=False)
    # Exclude the first neighbor (the point itself) and then predict values
    train_pred_data = train_dataset[train_neighbors[:, 1:]]
    # Predict the mean value among the k-nearest neighbors
    train_pred = np.mean(train_pred_data, axis=1)

    # Compute R2 score for the training data
    train_score = r2_score(train_pred, train_dataset).item()

    # Compute R2 score for the testing data
    test_score = r2_score(test_pred, test_dataset).item()

    if verbose:
        print("Train R2 Score:", train_score)
        print("Test R2 Score:", test_score)

    return train_score, test_score


def knn_classification(
    train_labels, train_embedding, test_labels, test_embedding, k=1, verbose=True, metric='minkowski'
):
    if verbose:
        print("Fitting k-NN classifier...")

    # Initialize the k-NN classifier for test set with n_neighbors=k
    knn_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric=metric)
    knn_classifier.fit(train_embedding, train_labels)

    if verbose:
        print("Making predictions on testing data...")

    # Make predictions on the testing data
    test_pred = knn_classifier.predict(test_embedding)

    # Re-fit the k-NN classifier for training set with n_neighbors=k+1
    knn_classifier.set_params(n_neighbors=k + 1)
    knn_classifier.fit(train_embedding, train_labels)

    if verbose:
        print("Making predictions on training data...")

    # Compute the (k + 1) nearest neighbors for the training data
    train_neighbors = knn_classifier.kneighbors(train_embedding, return_distance=False)
    # Exclude the first neighbor (the point itself) and then predict labels
    train_pred_labels = train_labels[train_neighbors[:, 1:]]
    # Predict the most frequent label among the k-nearest neighbors
    train_pred = mode(train_pred_labels, axis=1)[0]

    # Compute performance metrics for the training data
    train_acc = accuracy_score(train_labels, train_pred)
    train_f1 = f1_score(train_labels, train_pred, average="weighted")
    train_precision = precision_score(train_labels, train_pred, average="weighted")
    train_recall = recall_score(train_labels, train_pred, average="weighted")

    # Compute performance metrics for the testing data
    test_acc = accuracy_score(test_labels, test_pred)
    test_f1 = f1_score(test_labels, test_pred, average="weighted")
    test_precision = precision_score(test_labels, test_pred, average="weighted")
    test_recall = recall_score(test_labels, test_pred, average="weighted")

    if verbose:
        print("Train Accuracy:", train_acc)
        print("Test Accuracy:", test_acc)
        print("Train Precision:", train_precision)
        print("Test Precision:", test_precision)
        print("Train Recall:", train_recall)
        print("Test Recall:", test_recall)

    return (
        train_acc,
        train_f1,
        train_precision,
        train_recall,
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
    n_jobs=-1,
    verbose=True,
):
    # Step 1: Initialize the Model
    model = LinearRegression(n_jobs=n_jobs)

    r2_score = R2Score()
    r2_score.fit(train_dataset)

    # Step 2: Preprocessing
    # Scale the embeddings
    scaler = StandardScaler().fit(train_embedding)
    normalized_train_embedding = scaler.transform(train_embedding)
    normalized_test_embedding = scaler.transform(test_embedding)

    # Step 3: Train the Model
    # Train the model with a single call to `fit`
    model.fit(normalized_train_embedding, train_dataset)

    # Step 4: Evaluate the Model

    # Evaluation on train data
    train_pred = model.predict(normalized_train_embedding)
    train_r2_score = r2_score(train_pred, train_dataset).item()

    # Evaluation on test data
    test_pred = model.predict(normalized_test_embedding)
    test_r2_score = r2_score(test_pred, test_dataset).item()

    if verbose:
        print("Train R2 Score:", train_r2_score)
        print("Test R2 Score:", test_r2_score)

    return train_r2_score, test_r2_score


def linear_classification(
    train_labels, train_embedding, test_labels, test_embedding, verbose=True
):
    # Step 1: Initialize the Model
    model = LogisticRegression(max_iter=2000, n_jobs=-1, verbose=verbose)

    # Step 2: Preprocessing
    # Scale the embeddings
    scaler = StandardScaler().fit(train_embedding)
    normalized_train_embedding = scaler.transform(train_embedding)
    normalized_test_embedding = scaler.transform(test_embedding)

    # Step 3: Train the Model
    # Train the model with a single call to `fit`
    model.fit(normalized_train_embedding, train_labels)

    # Step 4: Evaluate the Model

    # Evaluation on train data
    y_pred_train = model.predict(normalized_train_embedding)
    f1_train = f1_score(train_labels, y_pred_train, average="weighted")
    acc_train = accuracy_score(train_labels, y_pred_train)
    precision_train = precision_score(train_labels, y_pred_train, average="weighted")
    recall_train = recall_score(train_labels, y_pred_train, average="weighted")

    # Evaluation on test data
    y_pred_test = model.predict(normalized_test_embedding)
    f1_test = f1_score(test_labels, y_pred_test, average="weighted")
    acc_test = accuracy_score(test_labels, y_pred_test)
    precision_test = precision_score(test_labels, y_pred_test, average="weighted")
    recall_test = recall_score(test_labels, y_pred_test, average="weighted")

    if verbose:
        print("Train Accuracy:", acc_train)
        print("Test Accuracy:", acc_test)
        print("Train Precision:", precision_train)
        print("Test Precision:", precision_test)
        print("Train Recall:", recall_train)
        print("Test Recall:", recall_test)

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
