import pytest
import torch

try:
    import sklearn
    has_sklearn = True
except ImportError:
    has_sklearn = False

pytestmark = pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")

if has_sklearn:
    from selfsne.nn.embedding import (
        PositionEmbedding2d,
        PositionEmbedding1d,
        SinusoidalPositionalEmbedding,
        RotaryPositionalEmbedding,
        PatchEmbedding1d,
        PatchEmbedding2d,
    )


def test_position_embedding_2d():
    model = PositionEmbedding2d(embedding_dim=4, height=2, width=3)
    x = torch.randn(1, 2, 2, 3)
    out = model(x)
    assert out.shape == (1, 6, 2, 3)


def test_position_embedding_1d():
    model = PositionEmbedding1d(embedding_dim=8, num_positions=10)
    x = torch.randn(2, 5, 8)
    out = model(x)
    assert out.shape == x.shape


def test_sinusoidal_positional_embedding():
    model = SinusoidalPositionalEmbedding(model_dim=8)
    x = torch.randn(2, 4, 8)
    out = model(x)
    assert out.shape == x.shape


def test_rotary_positional_embedding():
    model = RotaryPositionalEmbedding(model_dim=8)
    x = torch.randn(2, 4, 8)
    out = model(x)
    assert out.shape == x.shape


def test_patch_embedding_1d():
    model = PatchEmbedding1d(seq_length=8, patch_size=2, embedding_dim=4, in_channels=3)
    x = torch.randn(2, 3, 8)
    out = model(x)
    assert out.shape == (2, 4, 4)


def test_patch_embedding_2d():
    model = PatchEmbedding2d(image_size=8, patch_size=2, embedding_dim=4, in_channels=3)
    x = torch.randn(2, 3, 8, 8)
    out = model(x)
    assert out.shape == (2, 16, 4)
