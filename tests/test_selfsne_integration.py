"""Integration tests for SelfSNE training loops and LikelihoodRatioEstimator."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator


def _synthetic_loader(batch_size=8, n_batches=4):
    data = torch.randn(n_batches * batch_size, 4)
    dataset = TensorDataset(data, data + 0.01)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.mark.parametrize(
    "target_encoder,context_encoder",
    [
        (nn.Linear(4, 2), None),
        (None, nn.Linear(4, 2)),
        (nn.Linear(4, 2), nn.Linear(4, 2)),
        (nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), nn.Linear(4, 2)),
    ],
)
def test_selfsne_instantiation_with_varied_encoders(target_encoder, context_encoder):
    loss = LikelihoodRatioEstimator(divergence="bce", kernel="normal", baseline=0)
    model = SelfSNE(
        loss=loss,
        target_encoder=target_encoder,
        context_encoder=context_encoder,
        target_tokenizer=nn.Identity(),
        context_tokenizer=nn.Identity(),
    )
    # Model should have trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 0
    # Forward should run
    x = torch.randn(4, 4)
    out = model(x)
    assert "context_embedding" in out
    assert "target_embedding" in out


@pytest.mark.parametrize("kernel", ["euclidean", "normal", "student_t", "cosine"])
@pytest.mark.parametrize("divergence", ["bce", "jsd", "kld", "rkld"])
@pytest.mark.parametrize("baseline", [0, "batch", "constant"])
def test_likelihood_ratio_estimator_variants(kernel, divergence, baseline):
    loss = LikelihoodRatioEstimator(kernel=kernel, divergence=divergence, baseline=baseline)
    context = torch.randn(8, 4)
    target = torch.randn(8, 4)
    # Use embeddings directly
    emb_context = torch.randn(8, 2)
    emb_target = torch.randn(8, 2)
    out = loss(
        context_embedding=emb_context,
        target_embedding=emb_target,
    )
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def _make_model(kernel="normal", divergence="bce", baseline=0):
    loss = LikelihoodRatioEstimator(kernel=kernel, divergence=divergence, baseline=baseline)
    return SelfSNE(
        loss=loss,
        target_encoder=nn.Linear(4, 2),
        context_encoder=nn.Linear(4, 2),
        target_tokenizer=nn.Identity(),
        context_tokenizer=nn.Identity(),
    )


@pytest.mark.parametrize("kernel", ["euclidean", "normal", "student_t", "cosine"])
@pytest.mark.parametrize("divergence", ["bce", "jsd", "kld"])
@pytest.mark.parametrize("baseline", [0, "batch"])
def test_training_loop_runs_small_synthetic(kernel, divergence, baseline):
    pl.seed_everything(42)
    model = _make_model(kernel=kernel, divergence=divergence, baseline=baseline)
    loader = _synthetic_loader(batch_size=8, n_batches=4)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    loss_val = trainer.callback_metrics.get("loss")
    assert loss_val is not None
    assert torch.isfinite(loss_val)


def test_selfsne_with_different_loss_configs():
    configs = [
        {"kernel": "euclidean", "divergence": "bce", "baseline": 0},
        {"kernel": "student_t", "divergence": "jsd", "baseline": "batch"},
        {"kernel": "cosine", "divergence": "kld", "baseline": "constant"},
    ]
    for cfg in configs:
        loss = LikelihoodRatioEstimator(**cfg)
        model = SelfSNE(
            loss=loss,
            target_encoder=nn.Linear(4, 3),
            target_tokenizer=nn.Identity(),
        )
        # Ensure instantiation works and forward runs
        x = torch.randn(4, 4)
        out = model(x)
        assert out["context_embedding"].shape == (4, 3)
        assert out["target_embedding"].shape == (4, 3)
