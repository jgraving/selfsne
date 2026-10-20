import math

import pytest
import torch

from selfsne.baselines import (
    AdditiveBaseline,
    LogAdditiveBaseline,
    BatchConditionalBaseline,
    ConstantBaseline,
    EmbeddingBaseline,
    MomentumBaseline,
    ParametricBaseline,
)
from selfsne.divergences import DIVERGENCES, AlphaNCE, binary_cross_entropy
from selfsne.kernels import PAIRWISE_KERNELS, ROWWISE_KERNELS
from selfsne.selfsne import get_lr_scheduler
from selfsne.utils import inverse_softplus


@pytest.mark.parametrize("name", list(ROWWISE_KERNELS))
def test_pairwise_kernels_match_broadcasted_values_and_gradients(name):
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    actual = PAIRWISE_KERNELS[name](x, y, 0.7)
    expected = ROWWISE_KERNELS[name](x[:, None], y[None], 0.7)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.sum(), (x, y), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (x, y))
    for a, b in zip(grad, expected_grad):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("name", list(DIVERGENCES))
def test_divergences_have_finite_values_and_gradients(name):
    pos = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    neg = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    a, r = DIVERGENCES[name](pos, neg)
    assert a.ndim == r.ndim == 0
    assert torch.isfinite(a + r)
    (a + r).backward()
    assert torch.isfinite(pos.grad).all()
    if name != "dr":
        assert torch.isfinite(neg.grad).all()


@pytest.mark.parametrize("alpha,name", [(0.0, "kld"), (0.5, "bce"), (1.0, "rkld")])
def test_alpha_nce_recovers_documented_special_cases(alpha, name):
    pos, neg = torch.tensor([0.2, 0.5]), torch.tensor([-0.2, -1.0])
    for actual, expected in zip(AlphaNCE(alpha)(pos, neg), DIVERGENCES[name](pos, neg)):
        torch.testing.assert_close(actual, expected)


def test_bce_remains_finite_at_extreme_logits():
    pos = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    neg = pos.detach().clone().requires_grad_()
    sum(binary_cross_entropy(pos, neg)).backward()
    assert torch.isfinite(pos.grad).all() and torch.isfinite(neg.grad).all()


@pytest.mark.parametrize("momentum", [0.0, 0.5, 0.9, 1.0])
def test_momentum_baseline_update_eval_and_gradient(momentum):
    baseline = MomentumBaseline(momentum).double()
    values = torch.tensor(
        [[0.0, math.log(3.0)]], dtype=torch.float64, requires_grad=True
    )
    result = baseline(neg_similarity=values)
    expected = torch.tensor(
        [math.log(momentum + (1 - momentum) * 2)], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(
        values.grad, torch.exp(values.detach() - result.detach()) / values.numel()
    )
    before = baseline.log_moving_average.clone()
    baseline.eval()
    torch.testing.assert_close(baseline(neg_similarity=values + 5), before)
    torch.testing.assert_close(baseline.log_moving_average, before)


def test_baseline_instances_do_not_share_activation_state():
    first, second = ParametricBaseline(), ParametricBaseline()
    first.eval()
    assert second.activation.training
    assert first.activation is not second.activation


@pytest.mark.parametrize("cls", [AdditiveBaseline, LogAdditiveBaseline])
def test_combined_baselines_forward_all_inputs(cls):
    inputs = dict(
        pos_similarity=torch.ones(2, 1),
        neg_similarity=torch.ones(2, 3),
        context_embedding=torch.ones(2, 2),
        baseline_embedding=torch.zeros(2, 2),
        kernel=lambda x, y, scale: -(x - y).square().sum(-1),
        kernel_scale=1.0,
    )
    modules = [BatchConditionalBaseline(), EmbeddingBaseline()]
    actual = cls(modules, mean=True)(**inputs)
    values = torch.stack([module(**inputs) for module in modules])
    expected = (
        values.mean(0)
        if cls is AdditiveBaseline
        else torch.logsumexp(values, 0) - math.log(2)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cls", [AdditiveBaseline, LogAdditiveBaseline])
def test_empty_combination_rejected(cls):
    with pytest.raises(ValueError):
        cls([])


def test_inverse_softplus_handles_large_values():
    x = torch.tensor([1e-5, 1.0, 1000.0], dtype=torch.float64)
    actual = inverse_softplus(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(torch.nn.functional.softplus(actual), x)


@pytest.mark.parametrize(
    "warmup,hold,cosine",
    [(0, 0, 0), (0, 0, 4), (0, 3, 4), (3, 0, 4), (3, 2, 4), (3, 0, 0)],
)
def test_schedule_handles_missing_phases(warmup, hold, cosine):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = get_lr_scheduler(optimizer, warmup, hold, cosine)
    rates = []
    for _ in range(warmup + hold + cosine + 4):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert all(0 <= rate <= 1 for rate in rates)
    if warmup:
        assert rates[0] == pytest.approx(1 / warmup)
        assert rates[warmup - 1] == pytest.approx(1)
    if cosine:
        assert rates[warmup + hold] == pytest.approx(1)
        assert rates[warmup + hold + cosine] == pytest.approx(0)
        assert rates[-1] == pytest.approx(0)
    else:
        assert rates[-1] == pytest.approx(1)


@pytest.mark.parametrize("duration", [-1, 0.5, True])
def test_invalid_schedule_durations(duration):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    with pytest.raises(ValueError):
        get_lr_scheduler(optimizer, duration, 0, 0)
