import pickle

import numpy as np
import pytest
import torch

from selfsne.data import (
    PairedDataset,
    MemMapSequence,
    HDF5,
    HDF5Sequence,
    TensorDataLoader,
    NumpyDataLoader,
    correlated_gaussian_sampler,
    CorrelatedGaussianDataLoader,
)


def test_paired_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        PairedDataset([1, 2], [1])


@pytest.mark.parametrize("window_size,expected_count", [(1, 5), (3, 3), (5, 1), (6, 0)])
def test_memmap_windows_include_last_valid_window(
    tmp_path, window_size, expected_count
):
    data = np.arange(10).reshape(5, 2)
    path = tmp_path / "data.npy"
    np.save(path, data)
    dataset = MemMapSequence(path, window_size)
    assert len(dataset) == expected_count
    if expected_count:
        np.testing.assert_array_equal(dataset[-1], data[-window_size:].T)
        np.testing.assert_array_equal(dataset[0], data[:window_size].T)
    with pytest.raises(IndexError):
        dataset[expected_count]


@pytest.mark.parametrize("window_size", [0, -1, 1.5, True])
def test_invalid_windows_rejected(tmp_path, window_size):
    path = tmp_path / "data.npy"
    np.save(path, np.zeros((5, 2)))
    with pytest.raises(ValueError):
        MemMapSequence(path, window_size)


@pytest.mark.parametrize(
    "loader_class,data",
    [
        (TensorDataLoader, torch.arange(10).reshape(5, 2)),
        (NumpyDataLoader, np.arange(10).reshape(5, 2)),
    ],
)
@pytest.mark.parametrize("drop_last,count", [(False, 3), (True, 2)])
def test_tensor_loaders_preserve_order_and_handle_remainders(
    loader_class, data, drop_last, count
):
    loader = loader_class(data, batch_size=2, shuffle=False, drop_last=drop_last)
    batches = list(loader)
    assert len(loader) == len(batches) == count
    actual = torch.cat(batches)
    expected = torch.as_tensor(data[:4] if drop_last else data).float()
    torch.testing.assert_close(actual, expected)


def test_gaussian_sampler_accepts_float_mi_and_preserves_global_rng():
    before = torch.random.get_rng_state().clone()
    first = correlated_gaussian_sampler(8, 1.0, dim=2, seed=42)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    torch.testing.assert_close(
        first, correlated_gaussian_sampler(8, 1.0, dim=2, seed=42)
    )
    assert first.shape == (8, 4)
    assert torch.isfinite(first).all()


def test_gaussian_loader_includes_remainder_and_refreshes_samples():
    loader = CorrelatedGaussianDataLoader(5, 1.0, dim=2, seed=42, batch_size=2)
    first, second = list(loader), list(loader)
    assert len(loader) == len(first) == 3
    assert first[-1].shape[0] == 1
    assert not torch.equal(torch.cat(first), torch.cat(second))


def test_hdf5_windows_and_worker_serialization(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    data = np.arange(10).reshape(5, 2)
    with h5py.File(path, "w") as f:
        f["x"] = data
    dataset = HDF5Sequence(path, "x", window_size=3)
    assert len(dataset) == 3
    np.testing.assert_array_equal(dataset[-1], data[-3:])
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.h5file is None
    np.testing.assert_array_equal(restored[0], data[:3])
    with pytest.raises(IndexError):
        dataset[3]
    dataset.close()
    restored.close()


def test_hdf5_close_can_be_called_twice(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.ones((2, 3))
    dataset = HDF5(path, "x")
    dataset[0]
    dataset.close()
    dataset.close()
    assert dataset.h5file is None
