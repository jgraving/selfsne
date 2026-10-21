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

import operator
import os

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "HDF5 datasets require the 'data' extra: pip install 'selfsne[data]'."
        ) from exc
    return h5py


def _validate_window(window_size):
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < 1
    ):
        raise ValueError("window_size must be a positive integer.")


def _window_index(idx, length):
    idx = operator.index(idx)
    if idx < 0:
        idx += length
    if not 0 <= idx < length:
        raise IndexError("Window index out of range.")
    return idx


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
        if len(dataset_a) != len(dataset_b):
            raise ValueError("Paired datasets must have the same length.")
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
        return torch.sqrt(-torch.expm1(-2.0 / dim * mi))

    if n_samples < 1 or dim < 1:
        raise ValueError("n_samples and dim must be positive.")
    mi = torch.as_tensor(
        mutual_information, dtype=torch.get_default_dtype(), device=device
    )
    if mi.ndim != 0 or not torch.isfinite(mi) or mi < 0:
        raise ValueError("mutual_information must be a finite, non-negative scalar.")
    correlation = mi_to_rho(dim, mi)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device or "cpu").manual_seed(seed)
    z = torch.randn(n_samples, 2 * dim, device=device, generator=generator)
    x, eps = torch.split(z, dim, dim=1)
    # Derive the conditional scale from MI directly: subtracting rho**2 from
    # one loses the noise term once the rounded correlation reaches one.
    y = correlation * x + torch.exp(-mi / dim) * eps

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
        **kwargs,
    ):
        self.n_samples = n_samples
        self.mutual_information = mutual_information
        self.dim = dim
        self.seed = seed
        self._epoch = 0
        self._started = False
        if kwargs.get("persistent_workers", False):
            raise ValueError(
                "CorrelatedGaussianDataLoader regenerates its data each epoch; persistent_workers must be False."
            )
        self.batch_size = batch_size
        self.device = device
        self.dataset = self.generate_samples()
        super().__init__(self.dataset, batch_size=batch_size, **kwargs)

    def generate_samples(self):
        seed = None if self.seed is None else self.seed + self._epoch
        samples = correlated_gaussian_sampler(
            self.n_samples, self.mutual_information, self.dim, seed, self.device
        )
        self._epoch += 1
        return samples

    def __iter__(self):
        if self._started:
            self.dataset.copy_(self.generate_samples())
        self._started = True
        yield from super().__iter__()


class TensorDataLoader(DataLoader):
    """A standard DataLoader over an owned float32 tensor.

    Supports the usual sampler, generator, worker, and collation options.
    """

    def __init__(
        self,
        dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        device=None,
        **kwargs,
    ):
        if isinstance(dataset, np.ndarray):
            # Establish ownership and normalize byte order/strides before
            # crossing into PyTorch, including read-only memory maps.
            owned = np.array(dataset, dtype=np.float32, order="C", copy=True)
            self.tensor_dataset = torch.from_numpy(owned).to(device=device)
        else:
            self.tensor_dataset = (
                torch.as_tensor(dataset)
                .detach()
                .to(device=device, dtype=torch.float32)
                .clone()
            )
        self.shuffle = shuffle
        super().__init__(
            self.tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )


class _NumpyDataset(Dataset):
    """Convert requested rows only, preserving out-of-core array access."""

    def __init__(self, array):
        self.array = array

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_mapping", None)
        array = self.array
        if (
            isinstance(array, np.memmap)
            and array.filename is not None
            # Private edits in a copy-on-write mapping are absent from disk.
            # Serialize their values instead of reopening the original file.
            and array.mode != "c"
            and (array.flags.c_contiguous or array.flags.f_contiguous)
        ):
            root = array
            while isinstance(root.base, np.ndarray):
                root = root.base
            offset = root.offset + array.ctypes.data - root.ctypes.data
            state["_mapping"] = dict(
                filename=str(array.filename),
                dtype=array.dtype,
                mode="r",
                offset=offset,
                shape=array.shape,
                order="F"
                if array.flags.f_contiguous and not array.flags.c_contiguous
                else "C",
            )
            state["array"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state.get("_mapping") is not None:
            self.array = np.memmap(**state["_mapping"])

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        # Convert only this row to native-endian, positive-stride storage.
        row = np.array(self.array[index], dtype=np.float32, order="C", copy=True)
        return torch.from_numpy(row)

    def __getitems__(self, indices):
        # DataLoader's batched fetch path avoids one NumPy read per row.
        rows = np.asarray(self.array[np.asarray(indices)], dtype=np.float32, order="C")
        batch = torch.from_numpy(rows)
        return list(batch.unbind(0))


class NumpyDataLoader(DataLoader):
    """Batch NumPy arrays or memory maps without copying the entire dataset.

    Supports the usual sampler, generator, worker, and collation options.
    """

    def __init__(
        self, dataset, batch_size=1024, shuffle=True, drop_last=False, **kwargs
    ):
        # Lightning reinstantiates loaders with their already-wrapped dataset
        # when it injects the prediction batch sampler.
        if isinstance(dataset, _NumpyDataset):
            dataset = dataset.array
        if not isinstance(dataset, np.ndarray):
            raise TypeError("dataset must be a NumPy array or memory map.")
        if dataset.ndim == 0:
            raise ValueError("dataset must have a sample dimension.")
        self.numpy_dataset = dataset
        self.shuffle = shuffle
        super().__init__(
            _NumpyDataset(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )


class MNIST(Dataset):
    def __init__(self, return_images=False):
        super().__init__()
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:
            raise ImportError(
                "MNIST requires the 'data' extra: pip install 'selfsne[data]'."
            ) from exc
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
        _validate_window(window_size)
        self.window_size = window_size

        # Attractors
        attractor_1 = np.array([np.sqrt(b * (r - 1)), np.sqrt(b * (r - 1)), r - 1])
        attractor_2 = np.array([-np.sqrt(b * (r - 1)), -np.sqrt(b * (r - 1)), r - 1])

        self.targets = np.zeros(num_steps + 1)

        for i, point in enumerate(self.data.T):
            dist_1 = np.linalg.norm(point - attractor_1)
            dist_2 = np.linalg.norm(point - attractor_2)

            if dist_1 < dist_2:
                self.targets[i] = 0  # closer to attractor 1
            else:
                self.targets[i] = 1  # closer to attractor 2

    def __len__(self):
        return max(0, self.data.shape[-1] - self.window_size + 1)

    def __getitem__(self, idx):
        idx = _window_index(idx, len(self))
        return self.data[:, idx : idx + self.window_size].astype(np.float32)


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
        idx = _window_index(idx, len(self))
        return self.projection[:, idx : idx + self.window_size]


class MemMapSequence(Dataset):
    def __init__(self, file, window_size=100):
        super().__init__()
        self.data = np.load(file, mmap_mode="r")
        _validate_window(window_size)
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, idx):
        idx = _window_index(idx, len(self))
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
        self._owner_pid = None
        with _h5py().File(self.file, "r") as h5file:
            self.len = len(h5file[self.key])

    def _file(self):
        pid = os.getpid()
        if self.h5file is not None and self._owner_pid != pid:
            self.close()
        if self.h5file is None:
            self.h5file = _h5py().File(self.file, "r")
            self._owner_pid = pid
        return self.h5file

    def __getitem__(self, index):
        return self._file()[self.key][index]

    def __len__(self):
        return self.len

    def close(self):
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
        self._owner_pid = None

    def __getstate__(self):
        # A spawned DataLoader worker opens its own handle, rather than trying
        # to pickle an open HDF5 file inherited from the parent.
        state = self.__dict__.copy()
        state["h5file"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self):
        if getattr(self, "h5file", None) is not None:
            self.h5file.close()


class HDF5Sequence(HDF5):
    def __init__(self, file, key, window_size=100):
        super().__init__(file, key)
        _validate_window(window_size)
        self.window_size = window_size

    def __len__(self):
        return max(0, self.len - self.window_size + 1)

    def __getitem__(self, idx):
        idx = _window_index(idx, len(self))
        return self._file()[self.key][idx : idx + self.window_size]
