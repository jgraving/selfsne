"""Maintainer acceptance gate for the supported single-device CPU workflow."""

import json
import platform
from importlib.metadata import version

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.data import NumpyDataLoader
from selfsne.losses import LikelihoodRatioEstimator


def _encoder():
    return nn.Sequential(nn.Linear(64, 32), nn.SELU(), nn.Linear(32, 2))


def _loss():
    return LikelihoodRatioEstimator(
        kernel="cauchy", divergence="bce", baseline="parametric"
    )


def _model(steps):
    torch.manual_seed(17)
    return SelfSNE(
        loss=_loss(),
        target_encoder=_encoder(),
        learning_rate=0.003,
        lr_scheduler=True,
        lr_warmup_steps=2,
        lr_cosine_steps=8 * steps - 2,
    )


class _FiniteTrainingLoss(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"]).all()


def _trainer(epochs):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision=32,
        max_epochs=epochs,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        callbacks=[_FiniteTrainingLoss()],
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def _held_out_loss(model, loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for context, target in loader:
            total += float(model.loss(**model((context, target)))["loss"]) * len(
                context
            )
    return total / len(loader.dataset)


def test_cpu_alpha_acceptance(tmp_path, record_testsuite_property):
    datasets = pytest.importorskip("sklearn.datasets")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Labels are never supplied to the model or used to construct the pairs.
        values = datasets.load_digits().data.astype(np.float32) / 16
        observations = torch.from_numpy(values)
        generator = torch.Generator().manual_seed(23)
        order = torch.randperm(len(values), generator=generator)
        targets = observations + 0.02 * torch.randn(
            observations.shape, generator=generator
        )
        paired = TensorDataset(observations[order], targets[order])
        train = DataLoader(torch.utils.data.Subset(paired, range(1408)), batch_size=128)
        held_out = DataLoader(
            torch.utils.data.Subset(paired, range(1408, len(values))), batch_size=128
        )
        continuous = _model(len(train))
        initial_loss = _held_out_loss(continuous, held_out)
        continuous.train()
        uninterrupted_trainer = _trainer(8)
        uninterrupted_trainer.fit(continuous, train)
        final_loss = _held_out_loss(continuous, held_out)
        assert np.isfinite([initial_loss, final_loss]).all()
        assert final_loss <= 0.95 * initial_loss, (initial_loss, final_loss)

        interrupted = _model(len(train))
        first_trainer = _trainer(4)
        first_trainer.fit(interrupted, train)
        checkpoint = tmp_path / "interrupted.ckpt"
        first_trainer.save_checkpoint(checkpoint)
        resumed = _model(len(train))
        resumed_trainer = _trainer(8)
        resumed_trainer.fit(resumed, train, ckpt_path=checkpoint)
        assert (
            resumed_trainer.global_step
            == uninterrupted_trainer.global_step
            == 8 * len(train)
        )
        resume_error = 0.0
        for key, expected in continuous.state_dict().items():
            actual = resumed.state_dict()[key]
            assert torch.isfinite(actual).all(), key
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            resume_error = max(resume_error, float((actual - expected).abs().max()))
        assert (
            resumed_trainer.lr_scheduler_configs[0].scheduler.last_epoch
            == uninterrupted_trainer.lr_scheduler_configs[0].scheduler.last_epoch
        )
        assert (
            resumed_trainer.optimizers[0].param_groups[0]["lr"]
            == uninterrupted_trainer.optimizers[0].param_groups[0]["lr"]
        )
        expected_state = uninterrupted_trainer.optimizers[0].state_dict()["state"]
        actual_state = resumed_trainer.optimizers[0].state_dict()["state"]
        optimizer_error = 0.0
        for parameter_id, state in expected_state.items():
            for key, expected in state.items():
                actual = actual_state[parameter_id][key]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
                optimizer_error = max(
                    optimizer_error, float((actual - expected).abs().max())
                )
        final_checkpoint = tmp_path / "completed.ckpt"
        resumed_trainer.save_checkpoint(final_checkpoint)
        restored = SelfSNE.load_from_checkpoint(
            final_checkpoint,
            loss=_loss(),
            target_encoder=_encoder(),
            map_location="cpu",
        ).eval()
        inference = NumpyDataLoader(values, batch_size=449, shuffle=False)
        batches = resumed_trainer.predict(restored, inference)
        assert batches[-1]["target_embedding"].shape == (1, 2)
        embeddings = np.concatenate([batch["target_embedding"] for batch in batches])
        assert embeddings.shape == (1797, 2)
        assert np.isfinite(embeddings).all()
        assert (embeddings.std(axis=0) > 1e-3).all()
        resumed.eval()
        with torch.no_grad():
            expected = resumed(observations)["target_embedding"].numpy()
        np.testing.assert_allclose(embeddings, expected, rtol=1e-5, atol=1e-5)
        prediction_error = float(np.max(np.abs(embeddings - expected)))
        np.save(tmp_path / "embeddings.npy", embeddings)
        report = {
            "status": "passed",
            "dataset": "digits",
            "shape": list(embeddings.shape),
            "initial_held_out_loss": initial_loss,
            "final_held_out_loss": final_loss,
            "max_resume_error": resume_error,
            "max_optimizer_error": optimizer_error,
            "max_prediction_error": prediction_error,
            "python": platform.python_version(),
            "dependencies": {
                name: version(name)
                for name in ("torch", "pytorch-lightning", "numpy", "scikit-learn")
            },
        }
        (tmp_path / "acceptance.json").write_text(json.dumps(report, indent=2))
        record_testsuite_property("cpu_alpha_acceptance", json.dumps(report))
        print("CPU_ALPHA_ACCEPTANCE " + json.dumps(report))
    finally:
        torch.set_num_threads(old_threads)
