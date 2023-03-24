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

from torch.nn import Module
from torch.utils.data import Dataset

from selfsne.kernels import PAIRWISE_KERNELS
import numpy as np

from typing import Tuple, List, Any, Union


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


class IndexParser(Module):
    """
    Parses the index from a batch of data and returns it.

    Args:
        None
    """

    def forward(
        self,
        batch: Union[
            Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor, torch.Tensor]
        ],
    ) -> torch.Tensor:
        """
        Extracts the index from a batch of data.

        Args:
            batch (Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor, torch.Tensor]]): The batch of data to parse.
                Should be a tuple or list containing two elements, where the first element is the
                data and the second element is the index.

        Returns:
            torch.Tensor: The index extracted from the batch of data.
        """
        _, index = batch
        return index


class Queue(Module):
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
        self.register_buffer(
            "queue", torch.zeros((queue_size, num_features), dtype=dtype)
        )
        self.max_size = queue_size
        self.queue_size = 0
        self.freeze_on_full = freeze_on_full

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds the given data to the queue buffer. If the buffer is full and `freeze_on_full` is True,
        the buffer will not be updated.

        Args:
            x (torch.Tensor): The data to add to the buffer.

        Returns:
            torch.Tensor: The current state of the buffer.
        """
        if self.freeze:
            return self.queue
        else:
            self.queue = torch.cat((x, self.queue))[: self.max_size]
            self.queue_size = np.minimum(self.queue_size + x.shape[0], self.max_size)
            return self.queue[: self.queue_size]

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples `n_samples` data points randomly from the buffer.

        Args:
            n_samples (int): The number of data points to sample from the buffer.

        Returns:
            torch.Tensor: A tensor containing the sampled data points.
        """
        assert n_samples <= self.queue_size
        sample_idx = torch.randint(
            self.queue_size, (n_samples,), device=self.queue.device
        )
        return self.queue[sample_idx]

    @property
    def full(self) -> bool:
        """
        Returns True if the buffer is full, otherwise False.

        Returns:
            bool: True if the buffer is full, otherwise False.
        """
        return self.queue_size == self.max_size

    @property
    def freeze(self) -> bool:
        """
        Returns True if the buffer is full and `freeze_on_full` is True, otherwise False.

        Returns:
            bool: True if the buffer is full and `freeze_on_full` is True, otherwise False.
        """
        return self.full and self.freeze_on_full


class NearestNeighborSampler(Module):
    """
    Performs nearest neighbor sampling on a given batch of data.

    Args:
        num_features (int): The number of features in each data point.
        queue_size (int): The maximum size of the buffer for the data points (default: 2 ** 15).
        kernel (str): The name of the kernel to use for calculating the distances (default: "euclidean").
        num_neighbors (int): The number of nearest neighbors to sample from (default: 1).
        freeze_queue_on_full (bool): Whether to stop updating the buffer once it is full (default: False).
        return_index (bool): Whether to return the indices of the nearest neighbors (default: False).
    """

    def __init__(
        self,
        num_features: int,
        queue_size: int = 2 ** 15,
        kernel: str = "euclidean",
        num_neighbors: int = 1,
        freeze_queue_on_full: bool = False,
        return_index: bool = False,
    ):
        super().__init__()
        self.data_queue = Queue(num_features, queue_size, freeze_queue_on_full)
        self.kernel = PAIRWISE_KERNELS[kernel]
        self.num_neighbors = num_neighbors
        self.return_index = return_index
        if self.return_index:
            self.index_queue = Queue(
                1, queue_size, freeze_queue_on_full, dtype=torch.long
            )

    @torch.no_grad()
    def forward(
        self,
        batch: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            List[torch.Tensor, torch.Tensor],
        ],
    ) -> torch.Tensor:
        """
        Returns the nearest neighbors to the input data.

        Args:
            batch (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor, torch.Tensor]])): The batch of data to sample from.
                Should be a tensor or a tuple of two tensors containing the data and their indices.

        Returns:
            torch.Tensor: A tensor containing the nearest neighbor samples.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, index = batch
        else:
            data = batch
            if self.return_index:
                warnings.warn(
                    "`return_index` is True, but index was not passed with batch."
                )

        if self.training:
            if self.return_index:
                index_queue = self.index_queue(index)

            data_queue = self.data_queue(data)
            distances = self.kernel(data, data_queue)
            if not self.data_queue.freeze:
                # set self distances to inf
                batch_index = torch.arange(np.min(distances.shape), device=data.device)
                distances[batch_index, batch_index] = np.inf

            num_neighbors = np.minimum(self.num_neighbors, data_queue.shape[0])
            _, knn_index = torch.topk(distances, num_neighbors, dim=-1)

            if self.num_neighbors > 1:
                idx = torch.arange(knn_index.shape[0], device=knn_index.device)
                jdx = torch.randint(
                    0, num_neighbors, (knn_index.shape[0],), device=knn_index.device
                )
                neighbor_index = knn_index[idx, jdx]
            else:
                neighbor_index = knn_index[:, 0]

            if self.return_index:
                return index_queue[neighbor_index]
            else:
                return data_queue[neighbor_index]

        elif self.return_index:
            return index
        else:
            return x


def random_sample_columns(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Randomly samples columns from the input tensor.

    Args:
        x (torch.Tensor): The input tensor to sample from.
        num_samples (int): The number of columns to sample.

    Returns:
        torch.Tensor: A tensor containing the sampled columns.
    """
    idx = torch.arange(x.shape[0], device=x.device)
    jdx = torch.randint(0, num_samples, (x.shape[0],), device=x.device)
    return x[idx, jdx]
