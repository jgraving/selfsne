import io

import pytest
import torch

from selfsne.nn.norm import InputNorm1d, InputNorm2d, InputNorm3d


LAYERS = [
    (InputNorm1d, ()),
    (InputNorm1d, (1,)),
    (InputNorm2d, (1, 1)),
    (InputNorm3d, (1, 1, 1)),
]


@pytest.mark.parametrize("factory,spatial", LAYERS)
@pytest.mark.parametrize("momentum", [None, 0.25])
def test_running_normalization_matches_known_statistics_and_input_gradient(
    factory, spatial, momentum
):
    layer = factory(2, eps=0.25, momentum=momentum).double()
    first = torch.tensor([[-1.0, 1.0], [1.0, 5.0]], dtype=torch.float64)
    first = first.reshape(2, 2, *spatial)
    layer(first)
    second = (first + 4).requires_grad_()
    result = layer(second)
    if momentum is None:
        mean = torch.tensor([2.0, 5.0], dtype=torch.float64)
        variance = torch.tensor([2.0, 8.0], dtype=torch.float64)
    else:
        mean = torch.tensor([1.0, 2.3125], dtype=torch.float64)
        variance = torch.tensor([1.4375, 4.0625], dtype=torch.float64)
    torch.testing.assert_close(layer.running_mean, mean)
    torch.testing.assert_close(layer.running_var, variance)
    assert layer.num_batches_tracked == 2
    scale = (variance + 0.25).sqrt().reshape(1, 2, *spatial)
    expected = (second.detach() - mean.reshape(1, 2, *spatial)) / scale
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(second.grad, scale.reciprocal().expand_as(second))


@pytest.mark.parametrize("factory,spatial", LAYERS)
def test_checkpoint_restores_singleton_evaluation_and_keeps_statistics_fixed(
    factory, spatial
):
    layer = factory(2).double()
    data = torch.tensor([[-1.0, 1.0], [1.0, 5.0]], dtype=torch.float64).reshape(
        2, 2, *spatial
    )
    layer(data)
    checkpoint = io.BytesIO()
    torch.save(layer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = factory(2).double().eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    x = data[:1].clone().requires_grad_()
    before = {key: value.clone() for key, value in restored.state_dict().items()}
    actual = restored(x)
    expected = torch.tensor([[-1.0, -2.0]], dtype=torch.float64).reshape_as(x)
    scale = torch.tensor(
        [2.0 + restored.eps, 8.0 + restored.eps], dtype=torch.float64
    ).sqrt()
    expected = expected / scale.reshape(1, 2, *spatial)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("factory,spatial", LAYERS)
def test_multiple_training_forwards_keep_their_own_gradient_scale(factory, spatial):
    layer = factory(2).double()
    first = torch.tensor([[-1.0, 1.0], [1.0, 5.0]], dtype=torch.float64).reshape(
        2, 2, *spatial
    )
    first.requires_grad_()
    result = layer(first)
    first_scale = (layer.running_var.detach().clone() + layer.eps).sqrt()
    second = (3 * first.detach()).requires_grad_()
    later = layer(second)
    second_scale = (layer.running_var.detach().clone() + layer.eps).sqrt()
    (result.sum() + later.sum()).backward()
    torch.testing.assert_close(
        first.grad, first_scale.reciprocal().reshape(1, 2, *spatial).expand_as(first)
    )
    torch.testing.assert_close(
        second.grad, second_scale.reciprocal().reshape(1, 2, *spatial).expand_as(second)
    )
