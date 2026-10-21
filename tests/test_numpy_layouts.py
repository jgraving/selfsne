import numpy as np
import pytest
import torch

from selfsne.data import NumpyDataLoader


@pytest.mark.parametrize(
    "layout",
    ["contiguous", "reverse_rows", "reverse_features", "big_endian", "fortran"],
)
@pytest.mark.parametrize("batch_size", [None, 3])
def test_numpy_loader_accepts_array_layouts_and_owns_returned_data(layout, batch_size):
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
    loader = NumpyDataLoader(values, batch_size=batch_size, shuffle=False)
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
