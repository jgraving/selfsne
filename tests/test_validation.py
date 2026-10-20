import pytest
import torch
from torch import nn

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator, classifier_metrics


@pytest.mark.parametrize(
    "option,value",
    [
        ("kernel", "typo"),
        ("baseline", "typo"),
        ("divergence", "typo"),
        ("kernel_scale", 0),
        ("kernel_scale", -1),
        ("kernel_scale", float("nan")),
        ("kernel_scale", float("inf")),
        ("metrics", ["typo"]),
    ],
)
def test_invalid_loss_configuration(option, value):
    with pytest.raises(ValueError):
        LikelihoodRatioEstimator(**{option: value})


@pytest.mark.parametrize("shape", [(0, 2), (1, 2), (2, 3, 4)])
def test_invalid_embedding_batches(shape):
    with pytest.raises(ValueError):
        LikelihoodRatioEstimator()(torch.randn(shape), torch.randn(shape))


@pytest.mark.parametrize("reference_shape", [(3, 0, 2), (2, 3, 2), (3, 2, 4), (3, 2)])
def test_invalid_reference_shape(reference_shape):
    with pytest.raises(ValueError, match="reference"):
        LikelihoodRatioEstimator()(
            torch.randn(3, 2), torch.randn(3, 2), torch.randn(reference_shape)
        )


def test_singleton_with_explicit_references_is_supported():
    out = LikelihoodRatioEstimator()(
        torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 3, 2)
    )
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("logit", [-100.0, 0.0, 100.0])
def test_classifier_metrics_are_finite_for_constant_predictions(logit):
    metrics = classifier_metrics(torch.full((8,), logit), torch.full((8,), logit))
    assert all(torch.isfinite(v) for v in metrics.values())


def test_all_metrics_includes_classifier_metrics_when_enabled():
    loss = LikelihoodRatioEstimator(metrics="all", classifier_metrics=True).eval()
    out = loss(torch.randn(3, 2), torch.randn(3, 2))
    assert {"accuracy", "precision", "npv"} <= out.keys()


def test_identity_tokenizer_is_the_default():
    model = SelfSNE(loss=LikelihoodRatioEstimator(), target_encoder=nn.Linear(4, 2))
    assert model(torch.randn(3, 4))["target_embedding"].shape == (3, 2)


def test_shared_parameters_are_optimized_exactly_once():
    encoder = nn.Linear(2, 2)
    model = SelfSNE(
        loss=LikelihoodRatioEstimator(),
        target_encoder=encoder,
        baseline_head=encoder,
        target_head=encoder,
    )
    optimizer = model.configure_optimizers()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
