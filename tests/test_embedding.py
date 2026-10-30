# -*- coding: utf-8 -*-
"""Tests for selfsne.nn.embedding module (besides what test_sequences.py covers)."""

import pytest
import torch
from torch import nn

from selfsne.nn.embedding import (
    PositionEmbedding2d,
    PositionEmbedding1d,
    SinusoidalPositionalEmbedding,
    RotaryPositionalEmbedding,
    PatchEmbedding1d,
    PatchEmbedding2d,
)


class TestPositionEmbedding2d:
    """Tests for PositionEmbedding2d module."""

    def test_output_shape(self):
        """PositionEmbedding2d should concatenate position embeddings along channel dim."""
        pe = PositionEmbedding2d(embedding_dim=4, height=8, width=8)
        x = torch.randn(2, 3, 8, 8)  # (batch, channels, height, width)
        out = pe(x)
        # Should concatenate along channel dim: 3 + 4 = 7
        assert out.shape == (2, 7, 8, 8)

    def test_positional_embedding_is_learned(self):
        """PositionEmbedding2d should have learnable parameters."""
        pe = PositionEmbedding2d(embedding_dim=4, height=8, width=8)
        assert isinstance(pe.height_embedding, nn.Parameter)
        assert isinstance(pe.width_embedding, nn.Parameter)

    def test_positional_embedding_shape(self):
        """Positional embedding should have correct dimensions."""
        pe = PositionEmbedding2d(embedding_dim=4, height=8, width=16)
        x = torch.randn(2, 3, 8, 16)
        out = pe(x)
        assert out.shape == (2, 7, 8, 16)

    def test_batch_independence(self):
        """Positional embedding should be applied independently per batch."""
        pe = PositionEmbedding2d(embedding_dim=4, height=4, width=4)
        x = torch.randn(3, 2, 4, 4)
        out = pe(x)
        assert out.shape == (3, 6, 4, 4)


class TestPositionEmbedding1d:
    """Tests for PositionEmbedding1d module."""

    def test_output_shape(self):
        """PositionEmbedding1d should add position embeddings (not concatenate)."""
        pe = PositionEmbedding1d(embedding_dim=8, num_positions=10)
        x = torch.randn(2, 5, 8)
        out = pe(x)
        # Should add along seq dim: same shape
        assert out.shape == (2, 5, 8)

    def test_positional_embedding_is_learned(self):
        """PositionEmbedding1d should have learnable parameters."""
        pe = PositionEmbedding1d(embedding_dim=8, num_positions=10)
        assert isinstance(pe.embedding, nn.Parameter)

    def test_truncates_for_short_sequences(self):
        """PositionEmbedding1d should handle sequences shorter than num_positions."""
        pe = PositionEmbedding1d(embedding_dim=8, num_positions=20)
        x = torch.randn(2, 5, 8)
        out = pe(x)
        assert out.shape == (2, 5, 8)

    def test_positional_embedding_added_not_replaced(self):
        """PositionEmbedding1d should add to input, not replace."""
        pe = PositionEmbedding1d(embedding_dim=8, num_positions=10)
        x = torch.ones(2, 5, 8)
        out = pe(x)
        # Output should differ from input (position embeddings added)
        assert not torch.allclose(out, x)

    def test_handles_longer_sequences(self):
        """PositionEmbedding1d should handle sequences up to num_positions."""
        pe = PositionEmbedding1d(embedding_dim=8, num_positions=10)
        x = torch.randn(2, 10, 8)
        out = pe(x)
        assert out.shape == (2, 10, 8)


class TestSinusoidalPositionalEmbedding:
    """Tests for SinusoidalPositionalEmbedding module."""

    def test_output_shape(self):
        """SinusoidalPositionalEmbedding should preserve input shape."""
        pe = SinusoidalPositionalEmbedding(8)
        x = torch.randn(2, 10, 8)
        out = pe(x)
        assert out.shape == (2, 10, 8)

    def test_rejects_invalid_model_dim(self):
        """SinusoidalPositionalEmbedding should reject invalid model_dim."""
        with pytest.raises(ValueError, match="positive"):
            SinusoidalPositionalEmbedding(0)
        with pytest.raises(ValueError, match="positive"):
            SinusoidalPositionalEmbedding(-1)

    def test_requires_matching_model_dim(self):
        """SinusoidalPositionalEmbedding should reject mismatched input dimensions."""
        pe = SinusoidalPositionalEmbedding(8)
        x = torch.randn(2, 10, 16)
        with pytest.raises(AssertionError):
            pe(x)

    def test_positional_pattern(self):
        """SinusoidalPositionalEmbedding should create position-dependent embeddings."""
        pe = SinusoidalPositionalEmbedding(8)
        x = torch.zeros(1, 5, 8)
        out = pe(x)
        # Different positions should have different embeddings
        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 1], out[0, 2])

    def test_batch_independent(self):
        """SinusoidalPositionalEmbedding should work independently per batch."""
        pe = SinusoidalPositionalEmbedding(8)
        x = torch.ones(2, 5, 8)
        out = pe(x)
        torch.testing.assert_close(out[0], out[1])


class TestRotaryPositionalEmbedding:
    """Tests for RotaryPositionalEmbedding module."""

    def test_output_shape(self):
        """RotaryPositionalEmbedding should preserve input shape."""
        pe = RotaryPositionalEmbedding(8)
        x = torch.randn(2, 10, 8)
        out = pe(x)
        assert out.shape == (2, 10, 8)

    def test_rejects_odd_model_dim(self):
        """RotaryPositionalEmbedding should reject odd model_dim."""
        with pytest.raises(ValueError, match="even"):
            RotaryPositionalEmbedding(3)
        with pytest.raises(ValueError, match="even"):
            RotaryPositionalEmbedding(7)

    def test_rejects_invalid_model_dim(self):
        """RotaryPositionalEmbedding should reject invalid model_dim."""
        with pytest.raises(ValueError, match="positive"):
            RotaryPositionalEmbedding(0)

    def test_position_zero_is_identity(self):
        """RotaryPositionalEmbedding should be identity at position 0."""
        pe = RotaryPositionalEmbedding(8)
        x = torch.randn(2, 5, 8)
        out = pe(x)
        torch.testing.assert_close(out[:, 0], x[:, 0])

    def test_norm_preserved(self):
        """RotaryPositionalEmbedding should preserve per-token norms."""
        pe = RotaryPositionalEmbedding(8)
        x = torch.randn(2, 10, 8)
        out = pe(x)
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))

    def test_requires_matching_model_dim(self):
        """RotaryPositionalEmbedding should reject mismatched input dimensions."""
        pe = RotaryPositionalEmbedding(8)
        x = torch.randn(2, 10, 16)
        with pytest.raises(AssertionError):
            pe(x)


class TestPatchEmbedding1d:
    """Tests for PatchEmbedding1d function."""

    def test_output_shape(self):
        """PatchEmbedding1d should produce (batch, num_patches, embedding_dim)."""
        pe = PatchEmbedding1d(seq_length=20, patch_size=4, embedding_dim=16, in_channels=2)
        x = torch.randn(2, 2, 20)
        out = pe(x)
        assert out.shape == (2, 5, 16)  # 20/4 = 5 patches

    def test_rejects_invalid_seq_length(self):
        """PatchEmbedding1d should reject sequence lengths not divisible by patch_size."""
        with pytest.raises(AssertionError, match="divisible"):
            PatchEmbedding1d(seq_length=21, patch_size=4, embedding_dim=16, in_channels=2)

    def test_is_sequential(self):
        """PatchEmbedding1d should return an nn.Sequential module."""
        pe = PatchEmbedding1d(seq_length=20, patch_size=4, embedding_dim=16, in_channels=2)
        assert isinstance(pe, nn.Sequential)


class TestPatchEmbedding2d:
    """Tests for PatchEmbedding2d function."""

    def test_output_shape(self):
        """PatchEmbedding2d should produce (batch, num_patches, embedding_dim)."""
        pe = PatchEmbedding2d(image_size=28, patch_size=7, embedding_dim=16, in_channels=3)
        x = torch.randn(2, 3, 28, 28)
        out = pe(x)
        assert out.shape == (2, 16, 16)  # (28/7)^2 = 16 patches

    def test_rejects_invalid_image_size(self):
        """PatchEmbedding2d should reject image sizes not divisible by patch_size."""
        with pytest.raises(AssertionError, match="divisible"):
            PatchEmbedding2d(image_size=27, patch_size=7, embedding_dim=16, in_channels=3)

    def test_is_sequential(self):
        """PatchEmbedding2d should return an nn.Sequential module."""
        pe = PatchEmbedding2d(image_size=28, patch_size=7, embedding_dim=16, in_channels=3)
        assert isinstance(pe, nn.Sequential)

    def test_default_channels(self):
        """PatchEmbedding2d should default to 3 input channels."""
        pe = PatchEmbedding2d(image_size=28, patch_size=7, embedding_dim=16)
        x = torch.randn(2, 3, 28, 28)
        out = pe(x)
        assert out.shape == (2, 16, 16)
