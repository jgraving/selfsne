import importlib

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from selfsne.prior import MixturePrior


@pytest.mark.parametrize(
    "module", ["selfsne.plot", "selfsne.callbacks", "selfsne.data"]
)
def test_optional_modules_import(module):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    pytest.importorskip("shapely")
    pytest.importorskip("skimage")
    pytest.importorskip("rich")
    importlib.import_module(module)


def test_mixture_prior_named_kernel_log_prob_and_checkpoint_state():
    prior = MixturePrior(num_features=2, num_components=3)
    values = prior.log_prob(torch.randn(5, 2))
    assert values.shape == (5, 1)
    assert torch.isfinite(values).all()
    assert "watershed_assignments" in prior.state_dict()


def test_mixture_prior_supports_lightning_two_training_hooks():
    prior = MixturePrior(num_features=2, num_components=3)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(prior, DataLoader(torch.zeros(2, 1), batch_size=1))
    assert torch.isfinite(trainer.callback_metrics["entropy_epoch"])
    assert prior.watershed_assignments.shape == (3,)
