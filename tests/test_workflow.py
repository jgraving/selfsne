import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator


def make_model(**kwargs):
    return SelfSNE(
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_tokenizer=nn.Identity(),
        target_encoder=nn.Linear(4, 2),
        **kwargs,
    )


@pytest.mark.parametrize(
    "embeddings,logits", [(True, False), (False, True), (True, True), (False, False)]
)
def test_prediction_options(embeddings, logits):
    model = make_model(return_embeddings=embeddings, return_logits=logits).eval()
    output = model.predict_step(torch.randn(8, 4), 0)
    assert ("target_embedding" in output) == embeddings
    assert ("pos_logits" in output) == logits
    for value in output.values():
        assert np.isfinite(value).all()
    if embeddings:
        assert output["target_embedding"].shape == (8, 2)
    if logits:
        assert output["pos_logits"].shape == (8, 1)
        assert output["neg_logits"].shape == (8, 7)


def test_cpu_training_and_checkpoint_roundtrip(tmp_path):
    pl.seed_everything(42)
    context = torch.randn(16, 4)
    loader = DataLoader(TensorDataset(context, context + 0.01), batch_size=8)
    model = make_model()
    before = model.target_encoder.weight.detach().clone()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    assert torch.isfinite(trainer.callback_metrics["loss"])
    assert not torch.equal(before, model.target_encoder.weight)
    assert model.target_encoder.weight.grad is not None
    assert torch.isfinite(model.target_encoder.weight.grad).all()
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    # Architecture modules are deliberately excluded from Lightning hyperparameters.
    restored = SelfSNE.load_from_checkpoint(
        path,
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_tokenizer=nn.Identity(),
        target_encoder=nn.Linear(4, 2),
    ).eval()
    model.eval()
    with torch.no_grad():
        expected = model(context)["target_embedding"]
        actual = restored(context)["target_embedding"]
    torch.testing.assert_close(actual, expected)


def test_pair_construction_matches_direct_cauchy_calculation():
    context = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    loss = LikelihoodRatioEstimator(baseline=0.0)
    positive, negative = loss.similarities(context, target, kernel_scale=1.0)
    expected = -torch.log1p(((context[:, None] - target[None, :]) ** 2).sum(-1) / 2)
    torch.testing.assert_close(positive[:, 0], expected.diagonal())
    torch.testing.assert_close(
        negative, expected[~torch.eye(3, dtype=torch.bool)].reshape(3, 2)
    )


def test_bce_value_and_gradient_direction():
    from selfsne.divergences import binary_cross_entropy

    positive = torch.tensor([[0.0, 1.0]], requires_grad=True)
    negative = torch.tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
    attraction, repulsion = binary_cross_entropy(positive, negative)
    # Independent direct expression, using moderate logits to avoid overflow.
    expected = (
        torch.log1p(torch.exp(-positive)).mean()
        + torch.log1p(torch.exp(negative)).mean()
    )
    torch.testing.assert_close(attraction + repulsion, expected)
    (attraction + repulsion).backward()
    assert (positive.grad < 0).all()  # Minimization raises positive scores.
    assert (negative.grad > 0).all()  # Minimization lowers reference scores.


def test_prediction_preserves_order_and_includes_final_singleton():
    model = make_model().eval()
    data = torch.randn(9, 4)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    output = trainer.predict(model, DataLoader(data, batch_size=4, shuffle=False))
    actual = np.concatenate([batch["target_embedding"] for batch in output])
    with torch.no_grad():
        expected = model(data)["target_embedding"].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_explicit_references_match_in_batch_reference_scores():
    loss = LikelihoodRatioEstimator(baseline=0.0)
    context, target = torch.randn(4, 2), torch.randn(4, 2)
    reference = torch.stack(
        [torch.cat([target[:i], target[i + 1 :]]) for i in range(4)]
    )
    implicit = loss.logits(context, target)
    explicit = loss.logits(context, target, reference)
    torch.testing.assert_close(implicit["pos_logits"], explicit["pos_logits"])
    torch.testing.assert_close(implicit["neg_logits"], explicit["neg_logits"])


def test_explicit_reference_and_baseline_heads_receive_gradients():
    model = SelfSNE(
        loss=LikelihoodRatioEstimator(baseline="embedding", divergence="bce"),
        target_encoder=nn.Linear(4, 3),
        context_encoder=nn.Linear(4, 3),
        target_head=nn.Linear(3, 2),
        context_head=nn.Linear(3, 2),
        baseline_head=nn.Linear(3, 2),
    )
    batch = {
        "context": torch.randn(4, 4),
        "target": torch.randn(4, 4),
        "reference": torch.randn(4, 3, 4),
    }
    out = model(batch)
    assert out["reference_embedding"].shape == (4, 3, 2)
    model.loss(**out)["loss"].backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_training_with_scheduler_updates_and_logs_learning_rate(optimizer):
    model = make_model(
        optimizer=optimizer,
        lr_scheduler=True,
        lr_warmup_steps=1,
        lr_target_steps=0,
        lr_cosine_steps=3,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randn(8, 4)), batch_size=4
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    assert torch.isfinite(trainer.callback_metrics["lr_scale"])
    assert torch.isfinite(trainer.callback_metrics["loss"])
