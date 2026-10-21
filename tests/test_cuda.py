"""Execute with -m cuda --require-cuda; CPU skips do not establish CUDA support."""

import copy
import io
import json
import time

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator
from selfsne.neighbors import NearestNeighborSampler
from selfsne.utils import shuffle_rows

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    "kernel",
    [
        "normal",
        "cauchy",
        "student_t",
        "inverse",
        "laplace",
        "inner_product",
        "von_mises",
        "wrapped_cauchy",
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_cuda_loss_matches_cpu_values_and_gradients(kernel, explicit):
    generator = torch.Generator().manual_seed(11)
    tensors = [
        torch.randn(6, 8, generator=generator),
        torch.randn(6, 8, generator=generator),
    ]
    if explicit:
        tensors.append(torch.randn(6, 3, 8, generator=generator))
    cpu = [value.requires_grad_() for value in tensors]
    gpu = [value.detach().cuda().requires_grad_() for value in tensors]
    cpu_loss = LikelihoodRatioEstimator(kernel=kernel, baseline=0, divergence="bce")
    gpu_loss = copy.deepcopy(cpu_loss).cuda()
    expected = cpu_loss(*cpu)["loss"]
    actual = gpu_loss(*gpu)["loss"]
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=2e-5)
    actual.backward()
    expected.backward()
    for x, y in zip(cpu, gpu):
        assert y.grad.device.type == "cuda" and torch.isfinite(y.grad).all()
        torch.testing.assert_close(y.grad.cpu(), x.grad, rtol=1e-3, atol=2e-5)


def test_cuda_queue_checkpoint_and_sampling_devices():
    sampler = NearestNeighborSampler(
        1, queue_size=2, return_index=True, freeze_queue_on_full=True
    ).cuda()
    sampler(
        (
            torch.tensor([[0.0], [10.0]], device="cuda"),
            torch.tensor([11, 22], device="cuda"),
        )
    )
    saved = io.BytesIO()
    torch.save(sampler.state_dict(), saved)
    saved.seek(0)
    restored = NearestNeighborSampler(
        1, queue_size=2, return_index=True, freeze_queue_on_full=True
    )
    restored.load_state_dict(torch.load(saved, weights_only=True, map_location="cpu"))
    restored.cuda().eval()
    result = restored(
        (
            torch.tensor([[1.0], [9.0]], device="cuda"),
            torch.tensor([33, 44], device="cuda"),
        )
    )
    assert result.device.type == "cuda"
    torch.testing.assert_close(result.cpu(), torch.tensor([11, 22]))
    assert restored.data_queue.queue_size == restored.index_queue.queue_size == 2


def test_shuffle_rows_uses_cuda_rng_without_consuming_cpu_rng():
    values = torch.arange(64, device="cuda").reshape(16, 4)
    cpu_state = torch.get_rng_state().clone()
    torch.cuda.manual_seed(31)
    first = shuffle_rows(values)
    torch.cuda.manual_seed(31)
    second = shuffle_rows(values)
    torch.testing.assert_close(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    torch.testing.assert_close(first.sort(dim=0).values, values.sort(dim=0).values)


def _model(width=256, embedding_dim=16, scheduler=False):
    torch.manual_seed(41)
    return SelfSNE(
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_encoder=nn.Sequential(
            nn.Linear(64, width),
            nn.SELU(),
            nn.Linear(width, width),
            nn.SELU(),
            nn.Linear(width, embedding_dim),
        ),
        learning_rate=0.001,
        lr_scheduler=scheduler,
        lr_warmup_steps=1,
        lr_cosine_steps=15,
    )


def _trainer(epochs, callback_type, precision="32-true"):
    return pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback_type()],
        deterministic=True,
    )


@pytest.mark.parametrize(
    "precision",
    ["32-true", "16-mixed", pytest.param("bf16-mixed", marks=pytest.mark.cuda_bf16)],
)
def test_cuda_scaled_training_checkpoint_prediction(
    tmp_path, record_testsuite_property, precision, finite_training_callback
):
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("This CUDA device does not support bfloat16.")
    from sklearn.datasets import load_digits

    source = torch.tensor(load_digits().data, dtype=torch.float32) / 16
    # Scale a real-data workload with independently perturbed digit observations.
    generator = torch.Generator().manual_seed(23)
    observations = source.repeat(16, 1)
    targets = observations + 0.02 * torch.randn(observations.shape, generator=generator)
    loader = DataLoader(
        TensorDataset(observations, targets), batch_size=512, pin_memory=True
    )
    model = _model(width=1024, embedding_dim=32)
    before = model.target_encoder[0].weight.detach().clone()
    trainer = _trainer(2, finite_training_callback, precision)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.fit(model, loader)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    assert not torch.equal(before, model.target_encoder[0].weight.detach().cpu())
    checkpoint = tmp_path / (precision + ".ckpt")
    trainer.save_checkpoint(checkpoint)
    restored_cpu = SelfSNE.load_from_checkpoint(
        checkpoint,
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_encoder=_model(width=1024, embedding_dim=32).target_encoder,
        map_location="cpu",
    ).eval()
    for name, value in restored_cpu.state_dict().items():
        torch.testing.assert_close(
            model.state_dict()[name].cpu(), value, rtol=0, atol=0
        )
    with torch.no_grad():
        expected = restored_cpu(source)["target_embedding"].numpy()
    # Check checkpoint transfer at the same arithmetic precision. Comparing
    # bfloat16 output against float32 elementwise confounds restoration with
    # rounding, especially for outputs near zero after cancelling large terms.
    model.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        full_precision = model(source.cuda())["target_embedding"].cpu().numpy()
    np.testing.assert_allclose(full_precision, expected, rtol=2e-4, atol=2e-4)
    prediction_loader = DataLoader(source, batch_size=449)
    batches = trainer.predict(model, prediction_loader)
    actual = np.concatenate([batch["target_embedding"] for batch in batches])
    assert actual.dtype == np.float32 and actual.shape == (1797, 32)
    assert batches[-1]["target_embedding"].shape == (1, 32)
    assert np.isfinite(actual).all() and (actual.std(0) > 1e-3).all()
    roundtrip = trainer.predict(restored_cpu, prediction_loader)
    again = np.concatenate([batch["target_embedding"] for batch in roundtrip])
    np.testing.assert_allclose(again, actual, rtol=1e-5, atol=1e-5)
    monitor = next(
        c for c in trainer.callbacks if isinstance(c, finite_training_callback)
    )
    report = {
        "status": "passed",
        "precision": precision,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "samples": len(observations),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": 2,
        "seconds": duration,
        "observations_per_second": 2 * len(observations) / duration,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cross_precision_max_prediction_error": float(
            np.max(np.abs(actual - expected))
        ),
        "checkpoint_same_precision_max_prediction_error": float(
            np.max(np.abs(again - actual))
        ),
        "float32_checkpoint_max_prediction_error": float(
            np.max(np.abs(full_precision - expected))
        ),
        "finite_optimizer_steps": monitor.finite_steps,
        "amp_skipped_optimizer_steps": monitor.skipped_steps,
    }
    record_testsuite_property("cuda_acceptance_" + precision, json.dumps(report))
    print("CUDA_ACCEPTANCE " + json.dumps(report))


def test_cuda_epoch_resume_matches_uninterrupted(tmp_path, finite_training_callback):
    generator = torch.Generator().manual_seed(9)
    values = torch.rand(512, 64, generator=generator)
    loader = DataLoader(
        TensorDataset(values, values + 0.01), batch_size=128, pin_memory=True
    )
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        continuous = _model(scheduler=True)
        full = _trainer(4, finite_training_callback)
        full.fit(continuous, loader)
        interrupted = _model(scheduler=True)
        first = _trainer(2, finite_training_callback)
        first.fit(interrupted, loader)
        checkpoint = tmp_path / "resume.ckpt"
        first.save_checkpoint(checkpoint)
        resumed = _model(scheduler=True)
        final = _trainer(4, finite_training_callback)
        final.fit(resumed, loader, ckpt_path=checkpoint)
        assert full.global_step == final.global_step
        assert (
            full.lr_scheduler_configs[0].scheduler.last_epoch
            == final.lr_scheduler_configs[0].scheduler.last_epoch
        )
        assert (
            full.optimizers[0].param_groups[0]["lr"]
            == final.optimizers[0].param_groups[0]["lr"]
        )
        for name, expected in continuous.state_dict().items():
            torch.testing.assert_close(
                resumed.state_dict()[name], expected, rtol=1e-5, atol=2e-6
            )
        for key, expected_state in full.optimizers[0].state_dict()["state"].items():
            for field, expected in expected_state.items():
                torch.testing.assert_close(
                    final.optimizers[0].state_dict()["state"][key][field],
                    expected,
                    rtol=1e-5,
                    atol=2e-6,
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
