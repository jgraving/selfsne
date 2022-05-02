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


class Queue(Module):
    def __init__(self, queue_size=2 ** 15):
        super(Queue, self).__init__()
        self.queue_size = queue_size

    @torch.no_grad()
    def forward(self, x):
        if not hasattr(self, "queue"):
            self.register_buffer("queue", x)
        else:
            self.queue = torch.cat((self.queue, x))[-self.queue_size :]
        return self.queue

    @torch.no_grad()
    def sample(self, n_samples):
        indices = torch.randint(
            self.queue.shape[0], (n_samples,), device=self.queue.device
        )
        return self.queue[indices]

    @property
    def full(self):
        return self.queue.shape[0] == self.queue_size


def inner_product(x, y):
    return -(x @ y.T)


def cross_entropy(x, y):
    return inner_product(x.log().softmax(-1), y.log().log_softmax(-1))


def cosine(x, y):
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return inner_product(x_norm, y_norm)


METRICS = {
    "euclidean": torch.cdist,
    "inner_product": inner_product,
    "cross_entropy": cross_entropy,
    "cosine": cosine,
}


class NearestNeighborSampler(Module):
    def __init__(self, queue_size=2 ** 15, metric="euclidean"):
        super().__init__()
        self.queue = Queue(queue_size)
        self.metric = METRICS[metric]

    @torch.no_grad()
    def forward(self, x):
        if self.training:
            queue = self.queue(x)
            values, indices = torch.topk(-self.metric(x, queue), 2, dim=-1)
            return queue[indices[:, -1]]
        else:
            return x
