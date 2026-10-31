"""Tests for LikelihoodRatioEstimator kernel/divergence/baseline combinations."""
import pytest
import torch

from selfsne.losses import LikelihoodRatioEstimator


KERNELS = ["euclidean", "normal", "student_t", "cauchy", "cosine", "laplace", "inverse", "inner_product"]
DIVERGENCES = ["bce", "jsd", "kld", "rkld", "dr", "wasserstein"]
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
