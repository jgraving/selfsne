import pickle
import warnings

import numpy as np
import pytest
import torch

from selfsne.data import NumpyDataLoader, TensorDataLoader


@pytest.mark.parametrize(
    "layout",
    ["contiguous", "reverse_rows", "reverse_features", "big_endian", "fortran"],
)
@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("loader_class", [NumpyDataLoader, TensorDataLoader])
def test_numpy_loader_accepts_array_layouts_and_owns_returned_data(
    layout, batch_size, loader_class
):
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    if layout == "reverse_rows":
        values = values[::-1]
    elif layout == "reverse_features":
        values = values[:, ::-1]
    elif layout == "big_endian":
        values = values.astype(">f4")
    elif layout == "fortran":
        values = np.asfortranarray(values)
    before = values.copy()
    loader = loader_class(values, batch_size=batch_size, shuffle=False)
    batches = list(loader)
    result = torch.stack(batches) if batch_size is None else torch.cat(batches)
    torch.testing.assert_close(result, torch.tensor(values.astype(np.float32).copy()))
    assert result.dtype == torch.float32
    batches[0].fill_(-100)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("batch_size", [None, 3])
def test_big_endian_memory_map_loads_requested_rows(tmp_path, batch_size):
    file = tmp_path / "data.npy"
    np.save(file, np.arange(20, dtype=">i4").reshape(5, 4))
    values = np.load(file, mmap_mode="r")
    loader = NumpyDataLoader(values, batch_size=batch_size, shuffle=False)
    batches = list(loader)
    result = torch.stack(batches) if batch_size is None else torch.cat(batches)
    torch.testing.assert_close(
        result, torch.arange(20, dtype=torch.float32).reshape(5, 4)
    )
    assert isinstance(loader.dataset.array, np.memmap)


def test_tensor_loader_owns_read_only_numpy_input():
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    values.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loader = TensorDataLoader(values, batch_size=3, shuffle=False)
    # Ownership is established during construction, before iteration.
    values.flags.writeable = True
    values.fill(-100)
    torch.testing.assert_close(
        torch.cat(list(loader)), torch.arange(20, dtype=torch.float32).reshape(5, 4)
    )


@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_copy_on_write_memmap_edits_survive_worker_serialization(tmp_path, layout):
    path = tmp_path / "array.npy"
    original = np.arange(80, dtype=np.float32).reshape(20, 4)
    np.save(path, np.asfortranarray(original) if layout == "F" else original)
    array = np.load(path, mmap_mode="c")
    if layout == "slice":
        array = array[2:15]
    array[0] = -123
    dataset = NumpyDataLoader(array, shuffle=False).dataset
    restored = pickle.loads(pickle.dumps(dataset))
    torch.testing.assert_close(restored[0], torch.full((4,), -123.0))
    torch.testing.assert_close(restored[-1], torch.tensor(array[-1]))
    # The edited values exist only in this mapping, never in the backing file.
    np.testing.assert_array_equal(np.load(path), original)
