import pytest
import torch
from torch import nn

from selfsne.prior import MixturePrior, NormalPrior


@pytest.fixture(params=[NormalPrior, MixturePrior])
def prior(request):
    kwargs = {"num_components": 3} if request.param is MixturePrior else {}
    return request.param(num_features=2, **kwargs)


@pytest.mark.parametrize("training", [False, True])
def test_rate_preserves_modes_and_frozen_parameters(prior, training):
    prior.train(training)
    prior.probe = nn.Sequential(nn.Dropout(), nn.Linear(2, 2))
    prior.probe.train(not training)
    prior.probe[0].train(training)
    next(prior.parameters()).requires_grad_(False)
    modes = [module.training for module in prior.modules()]
    flags = [parameter.requires_grad for parameter in prior.parameters()]
    x = torch.randn(4, 2, requires_grad=True)

    prior.rate(x).sum().backward()

    assert [module.training for module in prior.modules()] == modes
    assert [parameter.requires_grad for parameter in prior.parameters()] == flags
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is None for parameter in prior.parameters())


def test_rate_restores_state_when_log_prob_raises(prior, monkeypatch):
    prior.eval()
    next(prior.parameters()).requires_grad_(False)
    flags = [parameter.requires_grad for parameter in prior.parameters()]

    def fail(x):
        raise RuntimeError("failed evaluation")

    monkeypatch.setattr(prior, "log_prob", fail)
    with pytest.raises(RuntimeError, match="failed evaluation"):
        prior.rate(torch.randn(4, 2))

    assert not prior.training
    assert [parameter.requires_grad for parameter in prior.parameters()] == flags


def test_normal_rate_matches_density_and_input_gradient():
    prior = NormalPrior(num_features=2)
    x = torch.tensor([[0.2, -1.0]], requires_grad=True)
    result = prior.rate(x)
    torch.testing.assert_close(result, torch.distributions.Normal(0, 1).log_prob(x))
    result.sum().backward()
    torch.testing.assert_close(x.grad, -x.detach())
    assert prior.log_scale.grad is None


def test_rate_does_not_update_momentum_baseline():
    prior = MixturePrior(num_features=2, num_components=3, baseline="momentum")
    prior.train()
    before = prior.baseline.log_moving_average.clone()
    x = torch.randn(4, 2, requires_grad=True)
    prior.rate(x).sum().backward()
    torch.testing.assert_close(prior.baseline.log_moving_average, before)
    assert prior.training and prior.baseline.training
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is None for parameter in prior.parameters())
