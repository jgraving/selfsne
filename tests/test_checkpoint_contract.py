import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.data import NumpyDataLoader, TensorDataLoader
from selfsne.losses import LikelihoodRatioEstimator


def _encoder():
    return nn.Sequential(nn.Linear(4, 8), nn.SELU(), nn.Linear(8, 2))


def _loss(**kwargs):
    return LikelihoodRatioEstimator(divergence="bce", **kwargs)


@pytest.fixture
def checkpoint(tmp_path):
    model = SelfSNE(loss=_loss(), target_encoder=_encoder())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    trainer.fit(model, DataLoader(TensorDataset(values, values), batch_size=4))
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    return path, model


@pytest.mark.parametrize(
    "options",
    [
        {"kernel": "normal"},
        {"kernel_scale": 2},
        {"divergence": "kld"},
        {"baseline": "batch"},
        {"pos_as_neg": True},
    ],
)
def test_restore_rejects_changed_loss_configuration(checkpoint, options):
    path, _ = checkpoint
    loss_options = {"divergence": "bce", **options}
    with pytest.raises(ValueError, match="loss configuration"):
        SelfSNE.load_from_checkpoint(
            path,
            loss=LikelihoodRatioEstimator(**loss_options),
            target_encoder=_encoder(),
        )


def test_restore_requires_explicit_loss_and_encoder(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="loss module"):
        SelfSNE.load_from_checkpoint(path, target_encoder=_encoder())
    with pytest.raises(ValueError, match="encoder"):
        SelfSNE.load_from_checkpoint(path, loss=_loss())


def test_restore_rejects_unknown_checkpoint_format(checkpoint):
    path, _ = checkpoint
    saved = torch.load(path, weights_only=False)
    saved["selfsne"] = {"format_version": 999}
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checkpoint format"):
        SelfSNE.load_from_checkpoint(path, loss=_loss(), target_encoder=_encoder())


def test_legacy_checkpoint_without_metadata_still_restores(checkpoint):
    path, model = checkpoint
    saved = torch.load(path, weights_only=False)
    saved.pop("selfsne", None)
    torch.save(saved, path)
    restored = SelfSNE.load_from_checkpoint(
        path, loss=_loss(), target_encoder=_encoder()
    ).eval()
    model.eval()
    x = torch.randn(5, 4)
    torch.testing.assert_close(
        restored(x)["target_embedding"], model(x)["target_embedding"]
    )


def test_restore_rejects_incompatible_encoder_shape(checkpoint):
    path, _ = checkpoint
    with pytest.raises(RuntimeError, match="size mismatch"):
        SelfSNE.load_from_checkpoint(
            path,
            loss=_loss(),
            target_encoder=nn.Sequential(nn.Linear(4, 8), nn.SELU(), nn.Linear(8, 3)),
        )


@pytest.mark.parametrize("loader", [NumpyDataLoader, TensorDataLoader])
def test_supported_prediction_loaders_preserve_contract(checkpoint, loader):
    path, _ = checkpoint
    model = SelfSNE.load_from_checkpoint(
        path, loss=_loss(), target_encoder=_encoder()
    ).eval()
    values = np.arange(36, dtype=np.float32).reshape(9, 4) / 36
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    batches = trainer.predict(model, loader(values, batch_size=4, shuffle=False))
    assert batches[-1]["target_embedding"].shape == (1, 2)
    actual = np.concatenate([batch["target_embedding"] for batch in batches])
    with torch.no_grad():
        expected = model(torch.from_numpy(values))["target_embedding"].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_equivalent_builtin_alias_restores(checkpoint):
    path, model = checkpoint
    restored = SelfSNE.load_from_checkpoint(
        path,
        loss=LikelihoodRatioEstimator(divergence="binary"),
        target_encoder=_encoder(),
    ).eval()
    model.eval()
    x = torch.randn(5, 4)
    torch.testing.assert_close(
        restored(x)["target_embedding"], model(x)["target_embedding"]
    )


def test_custom_loss_configuration_remains_caller_owned(checkpoint):
    from selfsne.baselines import ConstantBaseline

    path, _ = checkpoint
    model = SelfSNE(
        loss=LikelihoodRatioEstimator(baseline=ConstantBaseline(0.5)),
        target_encoder=_encoder(),
    )
    saved = {
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": pl.__version__,
        "hyper_parameters": {},
    }
    model.on_save_checkpoint(saved)
    assert saved["selfsne"]["loss_configuration"] is None
    torch.save(saved, path)
    restored = SelfSNE.load_from_checkpoint(
        path,
        loss=LikelihoodRatioEstimator(baseline=ConstantBaseline(0.5)),
        target_encoder=_encoder(),
    ).eval()
    model.eval()
    x = torch.randn(5, 4)
    torch.testing.assert_close(
        restored(x)["target_embedding"], model(x)["target_embedding"]
    )
