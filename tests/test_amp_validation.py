"""Test the CUDA acceptance monitor with a real CPU GradScaler."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _batch(monitor, model, optimizer, trainer, overflow=False):
    optimizer.zero_grad()
    monitor.on_train_batch_start(trainer, model, None, 0)
    loss = model(torch.ones(2, 3)).square().mean()
    scaler = trainer.precision_plugin.scaler
    scaler.scale(loss).backward()
    if overflow:
        next(model.parameters()).grad.fill_(float("inf"))
    scaler.unscale_(optimizer)
    monitor.on_before_optimizer_step(trainer, model, optimizer)
    scaler.step(optimizer)
    scaler.update()
    monitor.on_train_batch_end(trainer, model, {"loss": loss.detach()}, None, 0)


def test_amp_monitor_accepts_protected_overflow_then_finite_updates(
    finite_training_callback,
):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))
    monitor = finite_training_callback()
    before = [p.detach().clone() for p in model.parameters()]
    _batch(monitor, model, optimizer, trainer, overflow=True)
    assert scaler.get_scale() == 4
    for actual, original in zip(model.parameters(), before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not optimizer.state  # The scaler skipped the optimizer entirely.
    _batch(monitor, model, optimizer, trainer)
    _batch(monitor, model, optimizer, trainer)
    monitor.on_train_end(trainer, model)
    assert monitor.skipped_steps == 1 and monitor.finite_steps == 2
    assert any(
        not torch.equal(p, original) for p, original in zip(model.parameters(), before)
    )


@pytest.mark.parametrize("use_scaler", [False, True])
def test_amp_monitor_rejects_unprotected_nonfinite_gradients(
    finite_training_callback, use_scaler
):
    model = nn.Linear(3, 2)
    scaler = torch.amp.GradScaler("cpu", init_scale=8) if use_scaler else None
    trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))
    monitor = finite_training_callback()
    monitor.on_train_batch_start(trainer, model, None, 0)
    next(model.parameters()).grad = torch.full_like(
        next(model.parameters()), float("inf")
    )
    monitor.on_before_optimizer_step(trainer, model, None)
    with pytest.raises(AssertionError):
        monitor.on_train_batch_end(trainer, model, {"loss": torch.tensor(1.0)}, None, 0)


def test_amp_monitor_rejects_a_run_that_only_skips_updates(finite_training_callback):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = SimpleNamespace(
        precision_plugin=SimpleNamespace(
            scaler=torch.amp.GradScaler("cpu", init_scale=8)
        )
    )
    monitor = finite_training_callback()
    _batch(monitor, model, optimizer, trainer, overflow=True)
    with pytest.raises(AssertionError):
        monitor.on_train_end(trainer, model)
