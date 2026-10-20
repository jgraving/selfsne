"""Analytic checks on distributions whose likelihood ratio is known exactly."""

import pytest
import torch

from selfsne.divergences import (
    DIVERGENCES,
    GENERATORS,
    AlphaNCE,
    ForwardNCE,
    ReverseNCE,
    generalized_jensen_shannon_f,
    noise_contrastive_estimation_f,
    reverse_noise_contrastive_estimation_f,
)

P_COUNTS = torch.tensor([1, 3, 6])
Q_COUNTS = torch.tensor([5, 2, 3])
P = P_COUNTS.double() / P_COUNTS.sum()
Q = Q_COUNTS.double() / Q_COUNTS.sum()
LOG_RATIO = (P / Q).log()


def exact_lower_bound(objective):
    # Repeated samples represent exact rational probabilities without sampling noise.
    pos = LOG_RATIO.repeat_interleave(P_COUNTS)
    neg = LOG_RATIO.repeat_interleave(Q_COUNTS)
    attraction, repulsion = objective(pos, neg)
    return -(attraction + repulsion)


@pytest.mark.parametrize(
    "name",
    [
        "kld",
        "rkld",
        "jsd",
        "bce",
        "jeffreys",
        "hellinger",
        "pearson",
        "neymann",
        "le_cam",
        "soft_rkld",
    ],
)
def test_variational_objective_equals_generator_at_true_ratio(name):
    expected = (Q * GENERATORS[name](LOG_RATIO)).sum()
    torch.testing.assert_close(exact_lower_bound(DIVERGENCES[name]), expected)


@pytest.mark.parametrize("k", [0.25, 1.0, 7.0])
@pytest.mark.parametrize(
    "objective,generator",
    [
        (ForwardNCE, noise_contrastive_estimation_f),
        (ReverseNCE, reverse_noise_contrastive_estimation_f),
    ],
)
def test_nce_generators_equal_their_objectives_at_true_ratio(k, objective, generator):
    expected = (Q * generator(LOG_RATIO, K=k)).sum()
    torch.testing.assert_close(exact_lower_bound(objective(k=k)), expected)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_generalized_js_generator_matches_weighted_kl_definition(alpha):
    mixture = alpha * P + (1 - alpha) * Q
    expected = (
        alpha * (P * (P / mixture).log()).sum()
        + (1 - alpha) * (Q * (Q / mixture).log()).sum()
    )
    actual = (Q * generalized_jensen_shannon_f(LOG_RATIO, alpha)).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        generalized_jensen_shannon_f(torch.zeros(1), alpha), torch.zeros(1)
    )


@pytest.mark.parametrize("name", list(GENERATORS))
def test_generators_are_convex_in_the_density_ratio(name):
    ratio = torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64, requires_grad=True)
    f = GENERATORS[name](ratio.log())
    first = torch.autograd.grad(f.sum(), ratio, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), ratio)[0]
    assert (second >= -1e-10).all(), (name, second)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"k": float("nan")},
        {"log_k": float("inf")},
        {"k": 1, "log_k": 0},
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": float("nan")},
    ],
)
def test_invalid_nce_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AlphaNCE(**kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "kld",
        "rkld",
        "jsd",
        "bce",
        "jeffreys",
        "hellinger",
        "pearson",
        "neymann",
        "le_cam",
        "soft_rkld",
    ],
)
def test_true_ratio_is_a_stationary_optimum_and_bounds_other_scores(name):
    score = LOG_RATIO.clone().requires_grad_()
    attraction, repulsion = DIVERGENCES[name](
        score.repeat_interleave(P_COUNTS), score.repeat_interleave(Q_COUNTS)
    )
    loss = attraction + repulsion
    gradient = torch.autograd.grad(loss, score)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
    exact = (Q * GENERATORS[name](LOG_RATIO)).sum()
    for candidate in (torch.zeros_like(score), LOG_RATIO.flip(0), LOG_RATIO + 0.4):
        a, r = DIVERGENCES[name](
            candidate.repeat_interleave(P_COUNTS), candidate.repeat_interleave(Q_COUNTS)
        )
        assert -(a + r) <= exact + 1e-12


@pytest.mark.parametrize("objective", [ForwardNCE, ReverseNCE])
@pytest.mark.parametrize("k", [0.25, 1.0, 7.0])
def test_nce_optimum_is_not_shifted_by_noise_ratio(objective, k):
    score = LOG_RATIO.clone().requires_grad_()
    a, r = objective(k=k)(
        score.repeat_interleave(P_COUNTS), score.repeat_interleave(Q_COUNTS)
    )
    gradient = torch.autograd.grad(a + r, score)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "generator",
    [noise_contrastive_estimation_f, reverse_noise_contrastive_estimation_f],
)
@pytest.mark.parametrize("k", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_generator_noise_ratio_is_rejected(generator, k):
    with pytest.raises(ValueError):
        generator(LOG_RATIO, K=k)
