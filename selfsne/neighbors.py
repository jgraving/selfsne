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

import numpy as np


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        return data, index

    def __len__(self):
        return len(self.dataset)


class IndexParser(Module):
    def forward(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            _, index = batch
        else:
            raise InputError("index was not passed with batch.")
        return index


class Queue(Module):
    def __init__(
        self,
        num_features,
        queue_size=2 ** 15,
        freeze_on_full=False,
        dtype=torch.float32,
    ):
        super(Queue, self).__init__()
        self.register_buffer(
            "queue", torch.zeros((queue_size, num_features), dtype=dtype)
        )
        self.max_size = queue_size
        self.queue_size = 0
        self.freeze_on_full = freeze_on_full

    @torch.no_grad()
    def forward(self, x):
        if self.freeze:
            return self.queue
        else:
            self.queue = torch.cat((x, self.queue))[: self.max_size]
            self.queue_size = np.minimum(self.queue_size + x.shape[0], self.max_size)
            return self.queue[: self.queue_size]

    @torch.no_grad()
    def sample(self, n_samples):
        assert n_samples <= self.queue_size
        sample_idx = torch.randint(
            self.queue_size, (n_samples,), device=self.queue.device
        )
        return self.queue[sample_idx]

    @property
    def full(self):
        return self.queue_size == self.max_size

    @property
    def freeze(self):
        return self.full and self.freeze_on_full


def inner_product(x, y):
    return -(x @ y.T)


def cross_entropy(x, y):
    return inner_product(x.log().softmax(-1), y.log().log_softmax(-1))


def cosine(x, y, eps=1e-8):
    dot = inner_product(x, y)
    x_norm = x.norm(dim=-1, keepdim=True)
    y_norm = y.norm(dim=-1, keepdim=True)
    return dot / (x_norm * y_norm.T + eps)


METRICS = {
    "euclidean": torch.cdist,
    "inner_product": inner_product,
    "cross_entropy": cross_entropy,
    "cosine": cosine,
}


class NearestNeighborSampler(Module):
    def __init__(
        self,
        num_features,
        queue_size=2 ** 15,
        metric="euclidean",
        perplexity=1,
        freeze_queue_on_full=False,
        return_index=False,
    ):
        super().__init__()
        self.data_queue = Queue(num_features, queue_size, freeze_queue_on_full)
        self.metric = METRICS[metric]
        self.perplexity = perplexity
        self.return_index = return_index
        if self.return_index:
            self.index_queue = Queue(
                1, queue_size, freeze_queue_on_full, dtype=torch.long
            )

    @torch.no_grad()
    def forward(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, index = batch
        else:
            data = batch
            if self.return_index:
                raise InputError(
                    "`return_index` is True, but index was not passed with batch."
                )

        if self.training:
            if self.return_index:
                index_queue = self.index_queue(index)

            data_queue = self.data_queue(data)
            distances = self.metric(data, data_queue)
            if not self.data_queue.freeze:
                # set self distances to inf
                batch_index = torch.arange(np.min(distances.shape), device=data.device)
                distances[batch_index, batch_index] = np.inf

            perplexity = np.minimum(self.perplexity, data_queue.shape[0])
            _, knn_index = torch.topk(distances, perplexity, dim=-1, largest=False)

            if self.perplexity > 1:
                idx = torch.arange(knn_index.shape[0], device=knn_index.device)
                jdx = torch.randint(
                    0, perplexity, (knn_index.shape[0],), device=knn_index.device
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
