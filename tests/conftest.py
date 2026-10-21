"""CUDA checks skip on CPU CI, but are mandatory when explicitly requested."""

import os

import pytest
import pytorch_lightning as pl
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA hardware is available.",
    )


def pytest_configure(config):
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.getoption("--require-cuda"):
        if not torch.cuda.is_available():
            raise pytest.UsageError(
                "CUDA validation requires a CUDA-enabled PyTorch build and an accessible NVIDIA GPU; CPU fallback is disabled."
            )


def pytest_collection_modifyitems(config, items):
    cuda_items = [item for item in items if item.get_closest_marker("cuda")]
    if config.getoption("--require-cuda") and not cuda_items:
        raise pytest.UsageError(
            "No CUDA tests selected; CUDA validation cannot pass without executing them."
        )
    if not torch.cuda.is_available():
        for item in cuda_items:
            item.add_marker(pytest.mark.skip(reason="CUDA hardware is not available."))


@pytest.fixture
def finite_training_callback():
    class FiniteTraining(pl.Callback):
        """Allow AMP overflow only when the scaler protects an unchanged model."""

        def __init__(self):
            self.finite_steps = 0
            self.skipped_steps = 0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            scaler = getattr(trainer.precision_plugin, "scaler", None)
            self.scale_before = scaler.get_scale() if scaler is not None else None
            self.before_skip = None

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            if any(
                not torch.isfinite(p.grad).all()
                for p in pl_module.parameters()
                if p.grad is not None
            ):
                self.before_skip = [p.detach().clone() for p in pl_module.parameters()]

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            assert torch.isfinite(outputs["loss"]).all()
            for parameter in pl_module.parameters():
                assert torch.isfinite(parameter).all()
            overflow = any(
                not torch.isfinite(p.grad).all()
                for p in pl_module.parameters()
                if p.grad is not None
            )
            if overflow:
                scaler = getattr(trainer.precision_plugin, "scaler", None)
                assert scaler is not None and self.scale_before is not None
                assert scaler.get_scale() < self.scale_before
                assert self.before_skip is not None
                for actual, before in zip(pl_module.parameters(), self.before_skip):
                    torch.testing.assert_close(actual, before, rtol=0, atol=0)
                self.skipped_steps += 1
            else:
                self.finite_steps += 1

        def on_train_end(self, trainer, pl_module):
            # A run that never learns, or mostly overflows, must still fail.
            assert self.finite_steps > self.skipped_steps

    return FiniteTraining
