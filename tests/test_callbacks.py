# -*- coding: utf-8 -*-
"""Tests for selfsne.callbacks module."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar

from selfsne.callbacks import TotalStepsProgressBar


class SimpleModel(pl.LightningModule):
    """Minimal model for testing callbacks."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self(x).sum()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self(x).sum()
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.randn(16, 4), torch.randn(16, 2)), batch_size=4)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 2)), batch_size=4)


class TestTotalStepsProgressBar:
    """Tests for TotalStepsProgressBar callback."""

    def test_is_rich_progress_bar_subclass(self):
        """TotalStepsProgressBar should be a RichProgressBar subclass."""
        from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
        assert issubclass(TotalStepsProgressBar, RichProgressBar)

    def test_callback_tracks_global_step(self):
        """TotalStepsProgressBar should track global training steps."""
        callback = TotalStepsProgressBar()
        model = SimpleModel()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            callbacks=[callback],
        )
        trainer.fit(model)
        # Should have completed 2 epochs * 4 batches = 8 steps
        assert trainer.global_step == 8

    def test_callback_tracks_validation(self):
        """TotalStepsProgressBar should not crash during validation."""
        callback = TotalStepsProgressBar()
        model = SimpleModel()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            val_check_interval=1.0,
            callbacks=[callback],
        )
        # Should not raise
        trainer.fit(model)

    def test_progress_bar_appears(self):
        """TotalStepsProgressBar should create a progress bar."""
        callback = TotalStepsProgressBar()
        model = SimpleModel()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[callback],
        )
        trainer.fit(model)
        assert callback.train_progress_bar_id is not None

    def test_multiple_instances_are_independent(self):
        """Two TotalStepsProgressBar instances should have independent state."""
        callback1 = TotalStepsProgressBar()
        callback2 = TotalStepsProgressBar()
        assert callback1.train_progress_bar_id is None
        assert callback2.train_progress_bar_id is None
