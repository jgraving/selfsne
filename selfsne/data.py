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

# Lorenz attractor code modified from:
# https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

import h5py
from sklearn.datasets import fetch_openml


class PairedDataset(Dataset):
    """A dataset that pairs dataset_a and dataset_b together.

    Parameters
    dataset_a : iter
        The dataset_a to be paired with dataset_b.
    dataset_b : iter
        The dataset_b to be paired with dataset_a.
    shuffle : bool, optional
        Whether to shuffle the dataset_a and dataset_b before returning them in the getitem method. Default is False.

    """

    def __init__(self, dataset_a, dataset_b, shuffle=False):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.shuffle = shuffle
        self.dataset_a_indices = np.arange(len(dataset_a))
        self.dataset_b_indices = np.arange(len(dataset_b))
        if shuffle:
            np.random.shuffle(self.dataset_a_indices)
            np.random.shuffle(self.dataset_b_indices)

    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, idx):
        dataset_a_idx = self.dataset_a_indices[idx]
        dataset_b_idx = self.dataset_b_indices[idx]
        return self.dataset_a[dataset_a_idx], self.dataset_b[dataset_b_idx]


def correlated_gaussian_sampler(
    n_samples, mutual_information, dim=20, seed=None, device=None
):
    def mi_to_rho(dim, mi):
        return torch.sqrt(1 - torch.exp(-2.0 / dim * mi))

    correlation = mi_to_rho(dim, mutual_information)

    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(n_samples, 2 * dim, device=device)
    x, eps = torch.split(z, dim, dim=1)
    y = correlation * x + torch.sqrt(1.0 - correlation ** 2) * eps

    samples = torch.cat((x, y), dim=1)
    return samples


class CorrelatedGaussianDataLoader(DataLoader):
    def __init__(
        self,
        n_samples,
        mutual_information,
        dim=20,
        seed=None,
        batch_size=1024,
        device=None,
        **kwargs
    ):
        self.n_samples = n_samples
        self.mutual_information = mutual_information
        self.dim = dim
        self.seed = seed
        self.batch_size = batch_size
        self.device = device
        self.dataset = self.generate_samples()
        super().__init__(self.dataset, batch_size=batch_size, **kwargs)

    def generate_samples(self):
        samples = correlated_gaussian_sampler(
            self.n_samples, self.mutual_information, self.dim, self.seed, self.device
        )
        return samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for idx in range(len(self)):
            batch = self.dataset[idx * self.batch_size : (idx + 1) * self.batch_size]
            yield batch
        # regenerate the samples after each epoch
        self.dataset = self.generate_samples()


class TensorDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        device=None,
        **kwargs
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs
        )
        self.tensor_dataset = torch.tensor(dataset)
        if device is not None:
            self.tensor_dataset = self.tensor_dataset.to(device)
        self.shuffle = shuffle

    def __len__(self):
        if self.drop_last:
            return len(self.tensor_dataset) // self.batch_size
        else:
            return int(
                torch.ceil(torch.tensor(len(self.tensor_dataset) / self.batch_size))
            )

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(
                len(self.tensor_dataset), device=self.tensor_dataset.device
            )
        else:
            indices = torch.arange(
                len(self.tensor_dataset), device=self.tensor_dataset.device
            )
        for idx in range(len(self)):
            batch_indices = indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch = self.tensor_dataset[batch_indices]
            yield batch.float()


class NumpyDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size=1024, shuffle=True, drop_last=False, **kwargs
    ):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        self.numpy_dataset = dataset
        # self.batch_size = batch_size
        self.shuffle = shuffle
        # self.drop_last = drop_last
        self.indices = np.arange(len(dataset))

    def __len__(self):
        if self.drop_last:
            return len(self.numpy_dataset) // self.batch_size
        else:
            return int(np.ceil(len(self.numpy_dataset) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for idx in range(len(self)):
            batch_indices = self.indices[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            batch = torch.from_numpy(self.numpy_dataset[batch_indices]).float()
            yield batch


class MNIST(Dataset):
    def __init__(self, return_images=False):
        super().__init__()
        # Load data from https://www.openml.org/d/554
        self.x_train, self.y_train = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )
        if return_images:
            self.x_train = self.x_train.reshape((-1, 1, 28, 28))
        self.x_train /= 255

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return np.array(self.x_train[idx]).astype(np.float32)


def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


class Lorenz(Dataset):
    def __init__(self, window_size, dt=0.01, num_steps=200000, s=10, r=28, b=2.667):
        super().__init__()
        # Need one more for the initial values
        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)

        # Set initial values
        xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)

        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)

        self.data = np.stack((xs, ys, zs))
        self.window_size = window_size

    def __len__(self):
        return self.data.shape[-1] - self.window_size

    def __getitem__(self, idx):
        return self.data[:, idx : idx + self.window_size]


class LorenzRandomProjection(Lorenz):
    def __init__(
        self, window_size, x_dim, dt=0.01, num_steps=200000, s=10, r=28, b=2.667
    ):
        super().__init__(window_size, dt, num_steps, s, r, b)

        W = np.random.normal(scale=1 / np.sqrt(3), size=(3, x_dim))
        z = (self.data - self.data.mean(-1, keepdims=True)) / self.data.std(
            -1, keepdims=True
        )
        self.projection = np.random.normal(loc=W.T @ z, scale=1 / np.sqrt(x_dim))

    def __getitem__(self, idx):
        return self.projection[:, idx : idx + self.window_size]


class MemMapSequence(Dataset):
    def __init__(self, file, window_size=100):
        super().__init__()
        self.data = np.load(file, mmap_mode="r")
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return np.array(self.data[idx : idx + self.window_size]).T


class MemMap(Dataset):
    def __init__(self, file):
        super(MemMap, self).__init__()
        self.file = file
        self.data = None

    def __len__(self):
        return np.load(self.file, mmap_mode="r").shape[0]

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.load(self.file, mmap_mode="r")
        return np.array(self.data[idx])


class HDF5(Dataset):
    def __init__(self, file, key):
        self.file = file
        self.key = key
        self.h5file = None
        with h5py.File(self.file, "r") as h5file:
            self.len = len(h5file[self.key])

    def __getitem__(self, index):
        if self.h5file is None:
            self.h5file = h5py.File(self.file, "r")
        return self.h5file[self.key][index]

    def __len__(self):
        return self.len

    def __del__(self):
        if self.h5file is not None:
            self.h5file.close()


class HDF5Sequence(HDF5):
    def __init__(self, file, key, window_size=100):
        super().__init__(file, key)
        self.window_size = window_size

    def __len__(self):
        return self.len - self.window_size

    def __getitem__(self, idx):
        if self.h5file is None:
            self.h5file = h5py.File(self.file, "r")
        return self.h5file[self.key][idx : idx + self.window_size]
