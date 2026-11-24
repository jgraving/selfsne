"""Tests for LikelihoodRatioEstimator kernel/divergence/baseline combinations."""
import pytest
import torch

from selfsne.losses import LikelihoodRatioEstimator


KERNELS = ["euclidean", "normal", "student_t", "cauchy", "cosine", "laplace", "inverse", "inner_product"]
DIVERGENCES = ["bce", "jsd", "kld", "fkld", "rkld", "dr", "wasserstein"]
BASELINES = [0, 1.0, "batch", "constant", "momentum", "parametric"]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("divergence", DIVERGENCES)
@pytest.mark.parametrize("baseline", BASELINES)
def test_likelihood_ratio_estimator_kernel_divergence_baseline_combo(kernel, divergence, baseline):
    loss = LikelihoodRatioEstimator(kernel=kernel, divergence=divergence, baseline=baseline)
    context_emb = torch.randn(8, 4)
    target_emb = torch.randn(8, 4)
    out = loss(context_embedding=context_emb, target_embedding=target_emb)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    # Ensure loss is scalar
    assert out["loss"].dim() == 0


def test_likelihood_ratio_estimator_with_reference():
    loss = LikelihoodRatioEstimator(kernel="normal", divergence="bce", baseline=0)
    context_emb = torch.randn(4, 4)
    target_emb = torch.randn(4, 4)
    reference_emb = torch.randn(4, 3, 4)
    out = loss(
        context_embedding=context_emb,
        target_embedding=target_emb,
        reference_embedding=reference_emb,
    )
    assert torch.isfinite(out["loss"])


def test_likelihood_ratio_estimator_pos_as_neg():
    loss = LikelihoodRatioEstimator(kernel="cosine", divergence="jsd", baseline="batch", pos_as_neg=True)
    context_emb = torch.randn(4, 4)
    target_emb = torch.randn(4, 4)
    out = loss(context_embedding=context_emb, target_embedding=target_emb)
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("kernel", ["euclidean", "student_t", "cosine"])
def test_likelihood_ratio_estimator_gradients(kernel):
    loss = LikelihoodRatioEstimator(kernel=kernel, divergence="bce", baseline=0)
    context_emb = torch.randn(4, 2, requires_grad=True)
    target_emb = torch.randn(4, 2, requires_grad=True)
    out = loss(context_embedding=context_emb, target_embedding=target_emb)
    loss_val = out["loss"]
    loss_val.backward()
    assert context_emb.grad is not None
    assert target_emb.grad is not None
    assert torch.isfinite(context_emb.grad).all()
    assert torch.isfinite(target_emb.grad).all()


def test_likelihood_ratio_estimator_invalid_kernel():
    with pytest.raises(ValueError):
        LikelihoodRatioEstimator(kernel="invalid_kernel")


def test_likelihood_ratio_estimator_invalid_divergence():
    with pytest.raises(ValueError):
        LikelihoodRatioEstimator(divergence="invalid_divergence")


def test_likelihood_ratio_estimator_invalid_baseline_name():
    with pytest.raises(ValueError):
        LikelihoodRatioEstimator(baseline="invalid_baseline")


@pytest.mark.parametrize(
    "kernel", ["normal", "cauchy", "laplace", "cosine", "inner_product"]
)
def test_broadcast_reference_pool_matches_expanded_reference(kernel):
    """Shared (1, N, D) / (N, D) pools match the same references given as (B, N, D)."""
    loss = LikelihoodRatioEstimator(kernel=kernel, divergence="jsd", baseline="batch")
    context_emb = torch.randn(4, 3)
    target_emb = torch.randn(4, 3)
    pool_emb = torch.randn(5, 3)
    pool = pool_emb.unsqueeze(0)
    expanded = pool_emb.unsqueeze(0).expand(4, -1, -1)
    batched = loss.logits(context_emb, target_emb, expanded)
    broadcast = loss.logits(context_emb, target_emb, pool)
    torch.testing.assert_close(broadcast["pos_logits"], batched["pos_logits"])
    torch.testing.assert_close(broadcast["neg_logits"], batched["neg_logits"])
    assert broadcast["neg_logits"].shape == (4, 5)


def test_broadcast_reference_pool_training_loss_and_gradients():
    loss = LikelihoodRatioEstimator(kernel="normal", divergence="bce", baseline=0)
    context_emb = torch.randn(4, 2, requires_grad=True)
    target_emb = torch.randn(4, 2, requires_grad=True)
    pool_emb = torch.randn(6, 2, requires_grad=True)
    out = loss(
        context_embedding=context_emb,
        target_embedding=target_emb,
        reference_embedding=pool_emb.unsqueeze(0),
    )
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for name, grad in (("context", context_emb), ("target", target_emb), ("pool", pool_emb)):
        assert grad.grad is not None, name
        assert torch.isfinite(grad.grad).all(), name
        assert grad.grad.abs().sum() > 0, name


@pytest.mark.parametrize(
    "reference",
    [
        torch.randn(2, 5, 3),   # first dim neither 1 nor the batch size
        torch.randn(4, 0, 3),   # empty reference set
        torch.randn(4, 5, 2),   # feature mismatch
        torch.randn(5),         # unrecognizable shape
    ],
)
def test_broadcast_reference_embedding_invalid_shapes_raise(reference):
    loss = LikelihoodRatioEstimator()
    with pytest.raises(ValueError):
        loss.logits(torch.randn(4, 3), torch.randn(4, 3), reference)
