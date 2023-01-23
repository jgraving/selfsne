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
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree
from sklearn.metrics import r2_score, f1_score
from tqdm.autonotebook import tqdm

from selfsne.data import PairedDataset


class R2Score:
    def __init__(self):
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)

    def transform(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1, keepdims=True)
        ss_tot = np.sum((y_true - self.mean_) ** 2, axis=1, keepdims=True)
        r2 = 1 - (ss_res / ss_tot)
        return r2


class GibbsR2Score(R2Score):
    def __init__(self):
        super().__init__()

    def transform(self, y_true, y_pred):
        ss_res = np.sum((y_true * np.log(y_true) - y_true * np.log(y_pred)), axis=1)
        ss_tot = np.sum((y_true * np.log(y_true) - y_true * np.log(self.mean_)), axis=1)
        r2 = 1 - (ss_res / ss_tot)
        return r2


def calculate_pairwise_distance_correlation(
    dataset,
    embedding,
    batch_size=1000,
    correlation_type="spearman",
    verbose=True,
    shuffle=False,
):
    """Calculate the mean pairwise distance correlation between the original data and the embedding.

    Parameters
    ----------
    dataset : np.ndarray
        The original high dimensional data.
    embedding : np.ndarray
        The low dimensional embedding of the data.
    batch_size : int, optional
        The size of the batches to use when calculating the correlations. Default is 1000.
    correlation_type : str, optional
        The type of correlation to use. Can be 'spearman' or 'pearson'. Default is 'spearman'.
    verbose : bool, optional
        Whether to display a progress bar. Default is True.
    shuffle : bool, optional
        Whether to shuffle the data before creating the batches. Default is False.

    Returns
    -------
    float
        The mean pairwise distance correlation.
    """
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)
    correlations = []
    if verbose:
        dataloader = tqdm(dataloader)
    for data, embedding_batch in dataloader:
        data = data.numpy()
        data_distances = pdist(data)
        embedding_distances = pdist(embedding_batch)
        if correlation_type == "spearman":
            rho, p = spearmanr(data_distances, embedding_distances)
        elif correlation_type == "pearson":
            rho, p = pearsonr(data_distances, embedding_distances)
        correlations.append(rho)
    return np.mean(correlations)


def knn_probe_reconstruction(
    dataset,
    embedding,
    metric,
    k=1,
    batch_size=1000,
    verbose=True,
    shuffle=False,
    num_workers=1,
):
    """Calculate the mean R^2 score for a KNN model trained on high dimensional data and an embedding, assessing the ability of the embedding to reconstruct the data using the K nearest neighbors.

    Parameters
    ----------
    dataset : np.ndarray
        The high dimensional data.
    embedding : np.ndarray
        The low dimensional embedding of the data.
    metric : class
        Initialized class with transform method accepting y_true, y_pred
    k : int, optional
        The number of nearest neighbors to consider. Default is 10.
    batch_size : int, optional
        The batch size for training. Default is 1000.
    verbose : bool, optional
        Whether to display a progress bar during training. Default is True.
    shuffle : bool, optional
        Whether to shuffle the data and embedding before training. Default is False.

    Returns
    -------
    float
        The mean R^2 score of the KNN model.
    """
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    embedding_tree = KDTree(embedding)

    if verbose:
        prog_bar = tqdm(total=len(dataloader))
    knn_metric = []
    for data, embedding_batch in dataloader:
        data = data.numpy()
        embedding_batch = embedding_batch.numpy()
        knn_indices = embedding_tree.query(
            embedding_batch, k=k + 1, return_distance=False
        )[:, 1:]
        knn_data = dataset[knn_indices]
        knn = np.mean(knn_data, axis=1)
        knn_metric.append(metric.transform(data, knn))
        if verbose:
            prog_bar.update(1)
    return np.array(knn_metric)


def linear_probe_reconstruction(
    dataset,
    embedding,
    epochs=100,
    batch_size=1000,
    verbose=True,
    shuffle=False,
    tol=1e-10,
):
    """Calculate the mean R^2 score for a linear regression model trained on high dimensional data and an embedding, assessing the ability of the embedding to linearly reconstruct the data.

    Parameters
    ----------
    dataset : np.ndarray
        The high dimensional data.
    embedding : np.ndarray
        The low dimensional embedding of the data.
    epochs : int, optional
        The number of training epochs. Default is 5.
    batch_size : int, optional
        The batch size for training. Default is 1000.
    verbose : bool, optional
        Whether to display a progress bar during training. Default is True.
    shuffle : bool, optional
        Whether to shuffle the data and embedding before training. Default is False.
    tol : float, optional
        The tolerance threshold for early stopping. The training loop will stop if the change in mean loss between epochs is less than this threshold. Default is 1e-10.

    Returns
    -------
    float
        The mean R^2 score of the linear regression model.
    """
    model = nn.Linear(embedding.shape[1], dataset.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1)
    paired_dataset = PairedDataset(dataset, embedding, shuffle=shuffle)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)

    mean_loss_values = []
    for epoch in tqdm(range(epochs)):
        mean_loss = 0
        for data, embedding_batch in dataloader:
            data = data.float()
            embedding_batch = embedding_batch.float()
            optimizer.zero_grad()
            output = model(embedding_batch)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            mean_loss += loss
        mean_loss /= len(dataloader)
        mean_loss_values.append(mean_loss)

    r2_scores = []
    for data, embedding_batch in dataloader:
        data = data.float()
        embedding_batch = embedding_batch.float()
        model.eval()
        output = model(embedding_batch).detach().numpy()
        r2_scores.append(r2_score(data.detach().numpy(), output))

    return np.mean(r2_scores)


def linear_probe_classification_accuracy(
    embedding,
    labels,
    epochs=100,
    batch_size=1000,
    verbose=True,
    shuffle=False,
    tol=1e-10,
):
    """Calculate the classification accuracy for a linear classifier trained on an embedding and class labels.

    Parameters
    ----------
    labels : np.ndarray
        The class labels for the data.
    embedding : np.ndarray
        The low dimensional embedding of the data.
    epochs : int, optional
        The number of training epochs. Default is 5.
    batch_size : int, optional
        The batch size for training. Default is 1000.
    verbose : bool, optional
        Whether to display a progress bar during training. Default is True.
    shuffle : bool, optional
        Whether to shuffle the data and embedding before training. Default is False.
    tol : float, optional
        The tolerance threshold for early stopping. The training loop will stop if the change in mean loss between epochs is less than this threshold. Default is 1e-10.

    Returns
    -------
    float
        The classification accuracy of the linear classifier.
    """
    # Get number of classes
    num_classes = np.unique(labels).shape[0]

    # Define the model
    model = nn.Linear(embedding.shape[1], num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1)

    # Create a paired dataset of embedding and labels
    paired_dataset = PairedDataset(embedding, labels, shuffle=shuffle)

    # Create a data loader from the paired dataset
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)

    # Training loop
    for epoch in tqdm(range(epochs)):
        for embedding_batch, label_batch in dataloader:
            embedding_batch = embedding_batch.float()
            label_batch = label_batch.long()
            optimizer.zero_grad()
            output = model(embedding_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

    # Calculate the accuracy of the linear classifier on the entire dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for embedding_batch, label_batch in dataloader:
            embedding_batch = embedding_batch.float()
            label_batch = label_batch.long()
            output = model(embedding_batch)
            _, predicted = torch.max(output.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

    return correct / total


def linear_probe_classification_f1_score(
    embedding,
    labels,
    epochs=100,
    batch_size=1000,
    verbose=True,
    shuffle=False,
    tol=1e-10,
):
    """Calculate the F1 score for a linear classifier trained on an embedding and class labels.

    Parameters
        labels : np.ndarray
    The class labels for the data.
    embedding : np.ndarray
        The low dimensional embedding of the data.
    epochs : int, optional
        The number of training epochs. Default is 5.
    batch_size : int, optional
        The batch size for training. Default is 1000.
    verbose : bool, optional
        Whether to display a progress bar during training. Default is True.
    shuffle : bool, optional
        Whether to shuffle the data and embedding before training. Default is False.
    tol : float, optional
        The tolerance threshold for early stopping. The training loop will stop if the change in mean loss between epochs is less than this threshold. Default is 1e-10.

    Returns
    float
    The F1 score of the linear classifier.
    """
    # Get number of classes
    num_classes = np.unique(labels).shape[0]

    # Define the model
    model = nn.Linear(embedding.shape[1], num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1)

    # Create a paired dataset of embedding and labels
    paired_dataset = PairedDataset(embedding, labels, shuffle=shuffle)

    # Create a data loader from the paired dataset
    dataloader = DataLoader(paired_dataset, batch_size=batch_size)

    # Training loop
    for epoch in tqdm(range(epochs)):
        for embedding_batch, label_batch in dataloader:
            embedding_batch = embedding_batch.float()
            label_batch = label_batch.long()
            optimizer.zero_grad()
            output = model(embedding_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

    # Calculate the F1 score of the linear classifier on the entire dataset
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for embedding_batch, label_batch in dataloader:
            embedding_batch = embedding_batch.float()
            label_batch = label_batch.long()
            output = model(embedding_batch)
            _, predicted = torch.max(output.data, 1)
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(label_batch.tolist())

    return f1_score(true_labels, predicted_labels, average="weighted")
