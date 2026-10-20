import pytest
import torch
from selfsne.nn import MultiheadLinear, MultiheadMLP, Multilinear


@pytest.mark.parametrize("per_head", [False, True])
def test_multihead_linear_matches_independent_linear_layers(per_head):
    layer = MultiheadLinear(3, 4, 2).double()
    x = torch.randn(
        (5, 2, 3) if per_head else (5, 3), dtype=torch.float64, requires_grad=True
    )
    actual = layer(x)
    expected = torch.stack(
        [
            torch.nn.functional.linear(
                x[:, i] if per_head else x, layer.weight[..., i], layer.bias[0, i]
            )
            for i in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("batch_norm", [False, True])
def test_multihead_mlp_supports_training_and_evaluation(batch_norm):
    model = MultiheadMLP(
        3, 2, hidden_features=5, num_layers=2, num_heads=2, batch_norm=batch_norm
    )
    output = model(torch.randn(4, 3))
    assert output.shape == (4, 2, 2)
    output.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    assert model(torch.randn(1, 3)).shape == (1, 2, 2)


def test_multilinear_accepts_noncontiguous_sequences():
    model = Multilinear(4, order=2)
    x = torch.randn(2, 3, 4).transpose(0, 1).requires_grad_()
    actual = model(x)
    expected = model(x.contiguous())
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
