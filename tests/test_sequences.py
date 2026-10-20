import pytest
import torch
from torch import nn

from selfsne.nn import (
    CausalConv1d,
    SinusoidalPositionalEmbedding,
    RotaryPositionalEmbedding,
    TokenSampler,
)
from selfsne.nn.attention import Attention
from selfsne.transformers import TransformerEncoder


@pytest.mark.parametrize("kernel_size,dilation", [(1, 1), (3, 1), (3, 2)])
def test_causal_convolution_preserves_length_and_cannot_see_future(
    kernel_size, dilation
):
    conv = CausalConv1d(2, 3, kernel_size, dilation=dilation)
    x = torch.randn(2, 2, 12)
    changed = x.clone()
    changed[..., 7:] += 100
    out = conv(x)
    assert out.shape == (2, 3, 12)
    torch.testing.assert_close(out[..., :7], conv(changed)[..., :7])


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm", "dyt"])
def test_transformer_output_ignores_masked_input_tokens(norm):
    torch.manual_seed(42)
    encoder = TransformerEncoder(8, 2, 2, 1, norm_type=norm).eval()
    mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
    x = torch.randn(2, 4, 8)
    changed = x.clone()
    changed[mask] += 100
    out = encoder(x, mask)
    assert out.shape == (2, 1, 8)
    torch.testing.assert_close(out, encoder(changed, mask))


def test_fully_masked_attention_has_zero_output_and_finite_gradients():
    attention = Attention(4, 1)
    q = torch.randn(2, 1, 3, 4, requires_grad=True)
    k = torch.randn(2, 1, 3, 4, requires_grad=True)
    v = torch.randn(2, 1, 3, 4, requires_grad=True)
    out = attention(q, k, v, mask=torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(out, torch.zeros_like(out))
    out.sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))


def test_causal_attention_does_not_see_future_values():
    attention = Attention(4, 1, causal_mask=True)
    q, k, v = [torch.randn(1, 1, 5, 4) for _ in range(3)]
    changed = v.clone()
    changed[..., 3:, :] += 100
    torch.testing.assert_close(
        attention(q, k, v)[..., :3, :], attention(q, k, changed)[..., :3, :]
    )


@pytest.mark.parametrize("dim", [1, 3, 4, 7, 8])
def test_sinusoidal_positions_support_odd_dimensions(dim):
    x = torch.zeros(2, 5, dim, dtype=torch.float64)
    output = SinusoidalPositionalEmbedding(dim).double()(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    torch.testing.assert_close(output[0], output[1])
    torch.testing.assert_close(output[0, 0, 0::2], torch.zeros_like(output[0, 0, 0::2]))
    torch.testing.assert_close(output[0, 0, 1::2], torch.ones_like(output[0, 0, 1::2]))


def test_rotary_position_zero_is_identity_and_norm_is_preserved():
    x = torch.randn(2, 5, 8)
    output = RotaryPositionalEmbedding(8)(x)
    torch.testing.assert_close(output[:, 0], x[:, 0])
    torch.testing.assert_close(output.norm(dim=-1), x.norm(dim=-1))


def test_rotary_rejects_odd_dimension():
    with pytest.raises(ValueError, match="even"):
        RotaryPositionalEmbedding(3)


def test_fractional_token_sampling_keeps_at_least_one_token():
    x = torch.randn(2, 3, 4)
    sampler = TokenSampler(p=0.01)
    assert sampler(x).shape == (2, 1, 4)
    sampler.eval()
    torch.testing.assert_close(sampler(x), x)


@pytest.mark.parametrize("count", [0, -1, 0.5])
def test_invalid_token_counts_rejected(count):
    with pytest.raises(ValueError):
        TokenSampler(num_tokens=count)
