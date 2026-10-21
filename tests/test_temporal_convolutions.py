import pytest
import torch
from torch import nn

from selfsne.nn.conv import Conv1d, TCN, TCN2d, TCN3d


MODELS = [(TCN, ()), (TCN2d, (2,)), (TCN3d, (2, 2))]


@pytest.mark.parametrize("factory,spatial", MODELS)
@pytest.mark.parametrize("kernel", [2, 3, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_temporal_network_preserves_shape_and_backpropagates(
    factory, spatial, kernel, causal
):
    model = factory(
        2,
        3,
        hidden_channels=3,
        num_layers=2,
        n_blocks=1,
        kernel_size=kernel,
        causal=causal,
    )
    x = torch.randn(2, 2, *spatial, 9, requires_grad=True)
    result = model(x)
    assert result.shape == (2, 3, *spatial, 9)
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("factory,spatial", MODELS)
@pytest.mark.parametrize("shift", [False, True])
def test_causal_network_cannot_see_future_inputs(factory, spatial, shift):
    torch.manual_seed(3)
    model = factory(
        1,
        1,
        hidden_channels=2,
        num_layers=2,
        n_blocks=1,
        causal=True,
        causal_shift=shift,
    )
    x = torch.randn(1, 1, *spatial, 9)
    changed = x.clone()
    changed[..., 5:] += 100
    torch.testing.assert_close(model(x)[..., :5], model(changed)[..., :5])
    if shift:
        # With the shift, the output at t=5 cannot see the input at t=5 either.
        torch.testing.assert_close(model(x)[..., :6], model(changed)[..., :6])


@pytest.mark.parametrize("factory,spatial", MODELS)
def test_causal_network_rejects_batch_norm_that_leaks_future(factory, spatial):
    with pytest.raises(ValueError, match="batch_norm"):
        factory(1, 1, causal=True, batch_norm=True)


@pytest.mark.parametrize(
    "kernel,dilation,stride", [(2, 1, 1), (4, 1, 2), (2, 3, 2), (3, 2, 1)]
)
def test_same_convolution_matches_explicit_padding(kernel, dilation, stride):
    layer = Conv1d(1, 1, kernel_size=kernel, dilation=dilation, stride=stride).double()
    x = torch.randn(2, 1, 11, dtype=torch.float64)
    total = (kernel - 1) * dilation
    expected = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x, (total // 2, total - total // 2)),
        layer.weight,
        layer.bias,
        stride=stride,
        dilation=dilation,
    )
    torch.testing.assert_close(layer(x), expected)


@pytest.mark.parametrize("residual", [False, True])
def test_tcn3d_residual_flag_controls_skip_paths(residual):
    model = TCN3d(1, 1, hidden_channels=1, num_layers=1, n_blocks=1, residual=residual)
    convolutions = [
        module for module in model.modules() if isinstance(module, nn.Conv3d)
    ]
    with torch.no_grad():
        for conv in convolutions:
            conv.weight.zero_()
            conv.bias.zero_()
        convolutions[0].weight.fill_(1)
        convolutions[-1].weight.fill_(1)
    x = torch.ones(1, 1, 2, 2, 5)
    expected = 2 * torch.nn.functional.selu(x) if residual else torch.zeros_like(x)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("factory,spatial", MODELS)
def test_noncausal_batch_normalization_supports_training_and_evaluation(
    factory, spatial
):
    model = factory(
        2,
        3,
        hidden_channels=3,
        num_layers=2,
        n_blocks=1,
        kernel_size=4,
        batch_norm=True,
    )
    x = torch.randn(2, 2, *spatial, 9)
    result = model(x)
    assert result.shape == (2, 3, *spatial, 9)
    result.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    assert torch.isfinite(model(x[:1])).all()


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_odd_kernel_keeps_standard_convolution_padding_behavior(mode):
    reference = nn.Conv1d(2, 3, 3, dilation=2, stride=2, padding=2, padding_mode=mode)
    layer = Conv1d(2, 3, 3, dilation=2, stride=2, padding_mode=mode)
    layer.load_state_dict(reference.state_dict())
    x = torch.randn(2, 2, 11)
    torch.testing.assert_close(layer(x), reference(x))
