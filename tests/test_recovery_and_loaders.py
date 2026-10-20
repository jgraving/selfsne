"""Operational regressions: resumable training and standard DataLoader options."""

import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.data import (
    TensorDataLoader,
    NumpyDataLoader,
    CorrelatedGaussianDataLoader,
    HDF5,
)
from selfsne.losses import LikelihoodRatioEstimator
from selfsne.nn import CausalConv1d


@pytest.mark.parametrize("loader_class", [TensorDataLoader, NumpyDataLoader])
def test_custom_samplers_control_order_and_batch_count(loader_class):
    data = np.arange(10).reshape(5, 2)
    loader = loader_class(data, batch_size=2, shuffle=False, sampler=[4, 1, 3])
    batches = list(loader)
    assert len(loader) == len(batches) == 2
    torch.testing.assert_close(
        torch.cat(batches), torch.tensor(data[[4, 1, 3]]).float()
    )


@pytest.mark.parametrize("loader_class", [TensorDataLoader, NumpyDataLoader])
def test_custom_collation_is_applied(loader_class):
    data = np.arange(8).reshape(4, 2)
    loader = loader_class(
        data,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda rows: torch.stack(rows).sum(0),
    )
    batches = list(loader)
    torch.testing.assert_close(batches[0], torch.tensor(data[:2]).float().sum(0))


@pytest.mark.parametrize("loader_class", [TensorDataLoader, NumpyDataLoader])
def test_seeded_generator_controls_shuffling(loader_class):
    data = np.arange(40).reshape(20, 2)
    first = loader_class(
        data, batch_size=4, generator=torch.Generator().manual_seed(42)
    )
    second = loader_class(
        data, batch_size=4, generator=torch.Generator().manual_seed(42)
    )
    torch.testing.assert_close(torch.cat(list(first)), torch.cat(list(second)))


def test_gaussian_loader_refreshes_after_an_interrupted_epoch():
    loader = CorrelatedGaussianDataLoader(8, 1.0, dim=2, seed=42, batch_size=2)
    first = next(iter(loader))
    second = next(iter(loader))
    assert not torch.equal(first, second)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("length,kernel,dilation", [(12, 1, 1), (12, 3, 1), (11, 3, 2)])
def test_strided_causal_convolution_preserves_expected_length_and_causality(
    stride, length, kernel, dilation
):
    conv = CausalConv1d(2, 3, kernel_size=kernel, dilation=dilation, stride=stride)
    x = torch.randn(2, 2, length)
    changed = x.clone()
    changed[..., 7:] += 100
    result = conv(x)
    assert result.shape == (2, 3, math.ceil(length / stride))
    past = math.ceil(7 / stride)
    torch.testing.assert_close(result[..., :past], conv(changed)[..., :past])


def _trainer(max_steps, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def _model():
    return SelfSNE(
        target_encoder=nn.Linear(4, 2),
        loss=LikelihoodRatioEstimator(divergence="bce"),
        lr_scheduler=True,
        lr_warmup_steps=2,
        lr_cosine_steps=6,
    )


def test_epoch_boundary_resume_matches_an_uninterrupted_run(tmp_path):
    data = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    loader = DataLoader(TensorDataset(data, data + 0.01), batch_size=4, shuffle=False)
    pl.seed_everything(42)
    uninterrupted = _model()
    full = _trainer(-1, max_epochs=2)
    full.fit(uninterrupted, loader)
    pl.seed_everything(42)
    interrupted = _model()
    first_half = _trainer(-1, max_epochs=1)
    first_half.fit(interrupted, loader)
    checkpoint = tmp_path / "resume.ckpt"
    first_half.save_checkpoint(checkpoint)
    resumed = _model()
    second_half = _trainer(-1, max_epochs=2)
    second_half.fit(resumed, loader, ckpt_path=checkpoint)
    assert full.global_step == second_half.global_step == 4
    for actual, expected in zip(resumed.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(actual, expected)
    assert second_half.optimizers[0].param_groups[0]["lr"] == pytest.approx(
        full.optimizers[0].param_groups[0]["lr"]
    )
    assert (
        second_half.lr_scheduler_configs[0].scheduler.last_epoch
        == full.lr_scheduler_configs[0].scheduler.last_epoch
    )


def test_checkpoint_restores_momentum_baseline_for_scoring(tmp_path):
    model = SelfSNE(
        target_encoder=nn.Linear(4, 2),
        loss=LikelihoodRatioEstimator(divergence="bce", baseline="momentum"),
        return_logits=True,
    )
    data = torch.randn(8, 4)
    trainer = _trainer(2)
    trainer.fit(model, DataLoader(TensorDataset(data, data + 0.01), batch_size=4))
    checkpoint = tmp_path / "baseline.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = SelfSNE.load_from_checkpoint(
        checkpoint,
        target_encoder=nn.Linear(4, 2),
        loss=LikelihoodRatioEstimator(divergence="bce", baseline="momentum"),
    ).eval()
    model.eval()
    actual, expected = restored.predict_step(data, 0), model.predict_step(data, 0)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])


def test_hdf5_reopens_an_inherited_handle_in_a_new_process(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        handle["x"] = np.arange(8).reshape(4, 2)
    dataset = HDF5(path, "x")
    expected = dataset[0].copy()
    inherited = dataset.h5file
    original_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: original_pid + 1)
    np.testing.assert_array_equal(dataset[0], expected)
    assert dataset.h5file is not inherited
    dataset.close()


@pytest.mark.skipif(
    sys.platform.startswith("linux") and not Path("/proc/self/stat").exists(),
    reason="Worker process tests require /proc on Linux.",
)
@pytest.mark.parametrize("context", ["spawn", "fork"])
def test_hdf5_can_be_read_by_a_loader_worker(tmp_path, context):
    if context == "fork" and sys.platform != "linux":
        pytest.skip(
            "The fork worker path is exercised on Linux; other platforms use spawn."
        )
    if context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} is not available on this platform")
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    data = np.arange(16).reshape(8, 2)
    with h5py.File(path, "w") as handle:
        handle["x"] = data
    dataset = HDF5(path, "x")
    dataset[0]  # The parent has already opened a handle before workers start.
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
        multiprocessing_context=context,
        timeout=60,
    )
    actual = torch.cat(list(loader))
    torch.testing.assert_close(actual, torch.tensor(data))
    dataset.close()


@pytest.mark.skipif(
    sys.platform.startswith("linux") and not Path("/proc/self/stat").exists(),
    reason="Worker process tests require /proc on Linux.",
)
@pytest.mark.parametrize("loader_class", [TensorDataLoader, NumpyDataLoader])
def test_array_loaders_support_spawned_workers(loader_class):
    data = np.arange(16).reshape(8, 2)
    loader = loader_class(
        data,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        multiprocessing_context="spawn",
        timeout=60,
    )
    torch.testing.assert_close(torch.cat(list(loader)), torch.tensor(data).float())
