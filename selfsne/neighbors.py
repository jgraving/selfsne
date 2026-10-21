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
from torch.utils.data import Dataset

from selfsne.kernels import PAIRWISE_KERNELS

from typing import Tuple, List, Any, Union, Callable

NEG_INF = float("-inf")


class IndexedDataset(Dataset):
    """
    Wraps an existing dataset and returns both the data and its index when __getitem__ is called.

    Args:
        dataset (Dataset): The dataset to wrap.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Get the data and its index at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[Any, int]: A tuple containing the data and its index.
        """
        data = self.dataset[index]
        return data, index

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)


class IndexParser(nn.Module):
    """
    Parses the index from a batch of data and returns it.

    Args:
        None
    """

    def forward(
        self,
        batch: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Extracts the index from a batch of data.

        Args:
            batch (Union[Tuple[torch.Tensor], List[torch.Tensor]]): The batch of data to parse.
                Should be a tuple or list containing two elements, where the first element is the
                data and the second element is the index.

        Returns:
            torch.Tensor: The index extracted from the batch of data.
        """
        _, index = batch
        return index


class Queue(nn.Module):
    """
    A first-in-first-out (FIFO) queue.

    Args:
        num_features (int): The number of features in each data point that will be stored in the queue.
        queue_size (int): The maximum size of the queue (default: 2 ** 15).
        freeze_on_full (bool): Whether to stop updating the queue once it is full (default: False).
        dtype (torch.dtype): The data type of the queue tensor (default: torch.float32).
    """

    def __init__(
        self,
        num_features: int,
        queue_size: int = 2 ** 15,
        freeze_on_full: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super(Queue, self).__init__()
        if num_features <= 0 or queue_size <= 0:
            raise ValueError("num_features and queue_size must be positive.")
        self.register_buffer(
            "queue", torch.zeros((queue_size, num_features), dtype=dtype)
        )
        self.max_size = queue_size
        self.register_buffer("_queue_size", torch.zeros((), dtype=torch.long))
        self.freeze_on_full = freeze_on_full

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds the given data to the queue during training. If the queue is full and `freeze_on_full` is True,
        the queue will not be updated. During evaluation, the queue will not be updated.

        Args:
            x (torch.Tensor): The data to add to the queue.

        Returns:
            torch.Tensor: The current state of the queue.
        """
        if x.ndim != 2 or x.shape[1] != self.queue.shape[1]:
            raise ValueError("x must have shape (batch_size, num_features).")
        if not self.freeze and self.training:
            values = torch.cat((x.to(self.queue), self.queue))[: self.max_size]
            self.queue = values
            self._queue_size.fill_(min(self.queue_size + x.shape[0], self.max_size))
        return self.queue[: self.queue_size]

    @property
    def queue_size(self) -> int:
        """Number of valid rows, persisted alongside the queue in checkpoints."""
        return int(self._queue_size.item())

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples `n_samples` data points randomly from the queue.

        Args:
            n_samples (int): The number of data points to sample from the queue.

        Returns:
            torch.Tensor: A tensor containing the sampled data points.
        """
        if not 0 <= n_samples <= self.queue_size:
            raise ValueError("n_samples must be between zero and the current queue size.")
        if n_samples == 0:
            return self.queue[:0]
        sample_idx = torch.randint(
            self.queue_size, (n_samples,), device=self.queue.device
        )
        return self.queue[sample_idx]

    @property
    def full(self) -> bool:
        """
        Returns True if the queue is full, otherwise False.

        Returns:
            bool: True if the queue is full, otherwise False.
        """
        return self.queue_size == self.max_size

    @property
    def freeze(self) -> bool:
        """
        Returns True if the queue is full and `freeze_on_full` is True, otherwise False.

        Returns:
            bool: True if the queue is full and `freeze_on_full` is True, otherwise False.
        """
        return self.full and self.freeze_on_full


class NearestNeighborSampler(nn.Module):
    """
    Performs nearest neighbor sampling on a given batch of data.

    Args:
        num_features (int): The number of features in each data point.
        queue_size (int): The maximum size of the queue for the data points (default: 2 ** 15).
        kernel (Union[str, Callable]): The name of the kernel to use for calculating the similarity or a custom kernel function (default: "euclidean").
        num_neighbors (int): The number of nearest neighbors to sample from. Equivalent to perplexity from t-SNE (default: 1).
        freeze_queue_on_full (bool): Whether to stop updating the queue once it is full (default: False).
        return_index (bool): Whether to return the indices of the nearest neighbors instead of the data (default: False).
        max_similarity (float): The maximum similarity value to exclude from the selection when sampling neighbors. Defaults to 0.

    """

    def __init__(
        self,
        num_features: int,
        queue_size: int = 2 ** 15,
        kernel: Union[str, Callable] = "euclidean",
        num_neighbors: int = 1,
        freeze_queue_on_full: bool = False,
        return_index: bool = False,
        max_similarity: float = 0.0,
    ):
        super().__init__()
        self.data_queue = Queue(num_features, queue_size, freeze_queue_on_full)
        if isinstance(kernel, str):
            self.kernel = PAIRWISE_KERNELS[kernel]
        elif callable(kernel):
            self.kernel = kernel
        else:
            raise ValueError("Invalid kernel type. Expected str or callable.")
        if not isinstance(num_neighbors, int) or num_neighbors < 1:
            raise ValueError("num_neighbors must be a positive integer.")
        self.num_neighbors = num_neighbors
        self.return_index = return_index
        if self.return_index:
            self.index_queue = Queue(
                1, queue_size, freeze_queue_on_full, dtype=torch.long
            )
        self.max_similarity = max_similarity

    @torch.no_grad()
    def forward(
        self,
        batch: Union[
            torch.Tensor,
            Tuple[torch.Tensor],
            List[torch.Tensor],
        ],
    ) -> torch.Tensor:
        """
        Returns the nearest neighbors to the input data.

        Args:
            batch (Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]): The batch of data to sample from.
                Should be a tensor or a tuple of two tensors containing the data and their indices.

        Returns:
            torch.Tensor: A tensor containing the nearest neighbor samples.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, index = batch
        else:
            data = batch
            if self.return_index:
                raise ValueError(
                    "`return_index` is True, but index was not passed with batch."
                )

        if self.return_index:
            if index.ndim not in (1, 2) or index.shape[0] != data.shape[0] or (
                index.ndim == 2 and index.shape[1] != 1
            ):
                raise ValueError("index must have shape (batch_size,) or (batch_size, 1).")
            if index.dtype not in (torch.int32, torch.int64):
                raise ValueError("index must contain integers.")

        # Only a queue updated with this batch has corresponding diagonal entries.
        updated = self.data_queue.training and not self.data_queue.freeze
        data_queue = self.data_queue(data)
        if self.return_index:
            index_queue = self.index_queue(index.reshape(-1, 1))
        sample_idx = knn_sampler(
            data, data_queue, self.kernel, self.num_neighbors, self.max_similarity,
            exclude_diagonal=updated,
        )

        if self.return_index:
            selected = index_queue[sample_idx]
            return selected[:, 0] if index.ndim == 1 else selected
        return data_queue[sample_idx]


def knn_sampler(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel: callable,
    num_neighbors: int,
    max_similarity: float = 0.0,
    exclude_diagonal: bool = True,
) -> torch.Tensor:
    """
    Performs K-nearest neighbors sampling given two input tensors x1 and x2, a kernel function to compute the similarity
    between them, the number of neighbors to sample and an optional maximum similarity value to exclude from the selection.

    Args:
        x1 (torch.Tensor): The first input tensor of shape (batch_size, num_features).
        x2 (torch.Tensor): The second input tensor of shape (num_samples, num_features).
        kernel (callable): A callable function that computes the pairwise similarity between two tensors. It should take two
            arguments of shape (batch_size, num_features) and (num_samples, num_features) respectively and return a tensor
            of shape (batch_size, num_samples) representing the similarity between them.
        exclude_diagonal (bool): Exclude corresponding rows, for inputs sharing row identities.
            Set False for independent query and candidate sets. Defaults to True.
        num_neighbors (int): The number of nearest eligible neighbors to sample uniformly from.
        max_similarity (float): The maximum similarity value to exclude from the selection. Defaults to 0.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) representing the indices of the sampled neighbors for each batch element.
    """
    if not isinstance(num_neighbors, int) or num_neighbors < 1:
        raise ValueError("num_neighbors must be a positive integer.")
    batch_size, num_samples = x1.shape[0], x2.shape[0]
    if num_samples == 0:
        raise ValueError("No eligible neighbors: the candidate queue is empty.")
    similarity = kernel(x1, x2)
    if similarity.shape != (batch_size, num_samples):
        raise ValueError("kernel must return shape (batch_size, num_samples).")
    valid = torch.isfinite(similarity) & (similarity != max_similarity)
    if exclude_diagonal:
        diagonal_idx = torch.arange(min(batch_size, num_samples), device=similarity.device)
        valid[diagonal_idx, diagonal_idx] = False
    counts = valid.sum(dim=-1).clamp(max=num_neighbors)
    if (counts == 0).any():
        raise ValueError("No eligible neighbors for one or more input rows.")
    similarity = similarity.masked_fill(~valid, NEG_INF)
    knn_index = torch.topk(similarity, min(num_neighbors, num_samples), dim=-1).indices
    if num_neighbors == 1:
        return knn_index[:, 0]
    # Each row may have fewer eligible neighbors than the requested count.
    ranks = (torch.rand(batch_size, device=similarity.device) * counts).long()
    return knn_index[torch.arange(batch_size, device=similarity.device), ranks]
