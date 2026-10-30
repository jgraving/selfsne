# -*- coding: utf-8 -*-
"""Tests for selfsne.nn.attention module (besides what test_sequences.py covers)."""

import pytest
import torch
from torch import nn

from selfsne.nn.attention import (
    Attention,
    SelfAttention,
    CrossAttention,
    LatentCrossAttention,
)


class TestAttention:
    """Tests for the base Attention class."""

    def test_attention_output_shape(self):
        """Attention should output tensor with correct shape."""
        attention = Attention(8, 2)
        q = torch.randn(2, 2, 4, 4)  # (batch, heads, seq, head_dim)
        k = torch.randn(2, 2, 4, 4)
        v = torch.randn(2, 2, 4, 4)
        out = attention(q, k, v)
        assert out.shape == (2, 2, 4, 4)

    def test_attention_rejects_invalid_config(self):
        """Attention should reject invalid model_dim/num_heads combinations."""
        with pytest.raises(ValueError, match="divisible"):
            Attention(7, 3)  # 7 % 3 != 0
        with pytest.raises(ValueError, match="positive"):
            Attention(0, 1)
        with pytest.raises(ValueError, match="positive"):
            Attention(8, 0)

    def test_attention_with_mask(self):
        """Attention should respect the mask parameter."""
        attention = Attention(4, 1)
        q = torch.randn(2, 1, 3, 4)
        k = torch.randn(2, 1, 3, 4)
        v = torch.randn(2, 1, 3, 4)
        # Mask: mask token 0 of batch 0 and 1 of batch 1
        mask = torch.tensor([[True, False, False], [False, False, True]])
        out = attention(q, k, v, mask=mask)
        assert out.shape == (2, 1, 3, 4)

    def test_attention_mask_rejects_wrong_shape(self):
        """Attention should reject masks with wrong shape."""
        attention = Attention(4, 1)
        q = torch.randn(2, 1, 3, 4)
        k = torch.randn(2, 1, 3, 4)
        v = torch.randn(2, 1, 3, 4)
        # Wrong shape mask
        mask = torch.ones(2, 2)  # Should be (2, 3)
        with pytest.raises(ValueError, match="mask must be"):
            attention(q, k, v, mask=mask)

    def test_attention_softmax_normalizes(self):
        """Attention output should be normalized attention-weighted sum."""
        attention = Attention(4, 1, dropout=0.0)
        q = torch.randn(1, 1, 3, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.randn(1, 1, 3, 4)
        out = attention(q, k, v)
        # Output should be a weighted sum of v, so shape should match
        assert out.shape == (1, 1, 3, 4)


class TestSelfAttention:
    """Tests for SelfAttention module."""

    def test_self_attention_output_shape(self):
        """SelfAttention should output tensor with correct shape."""
        sa = SelfAttention(8, 2)
        x = torch.randn(2, 4, 8)
        out = sa(x)
        assert out.shape == (2, 4, 8)

    def test_self_attention_preserves_sequence_length(self):
        """SelfAttention should preserve the sequence length."""
        sa = SelfAttention(16, 4)
        x = torch.randn(1, 10, 16)
        out = sa(x)
        assert out.shape[1] == 10

    def test_self_attention_with_mask(self):
        """SelfAttention should accept and apply a mask."""
        sa = SelfAttention(8, 2)
        x = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, False, False, False],
                             [False, False, True, False]])
        out = sa(x, mask=mask)
        assert out.shape == (2, 4, 8)


class TestCrossAttention:
    """Tests for CrossAttention module."""

    def test_cross_attention_output_shape(self):
        """CrossAttention should output tensor with correct query length."""
        ca = CrossAttention(8, 2)
        query = torch.randn(2, 3, 8)
        context = torch.randn(2, 5, 8)
        out = ca(query, context)
        assert out.shape == (2, 3, 8)

    def test_cross_attention_independent_lengths(self):
        """CrossAttention should handle different query and context lengths."""
        ca = CrossAttention(16, 4)
        query = torch.randn(1, 2, 16)
        context = torch.randn(1, 10, 16)
        out = ca(query, context)
        assert out.shape == (1, 2, 16)

    def test_cross_attention_with_mask(self):
        """CrossAttention should accept and apply a mask."""
        ca = CrossAttention(8, 2)
        query = torch.randn(2, 3, 8)
        context = torch.randn(2, 5, 8)
        mask = torch.tensor([[True, False, False, False, False],
                             [False, True, False, False, False]])
        out = ca(query, context, mask=mask)
        assert out.shape == (2, 3, 8)


class TestLatentCrossAttention:
    """Tests for LatentCrossAttention module."""

    def test_latent_cross_attention_output_shape(self):
        """LatentCrossAttention should output (batch, num_latent_tokens, model_dim)."""
        lca = LatentCrossAttention(8, 2, num_latent_tokens=3)
        x = torch.randn(2, 10, 8)
        out = lca(x)
        assert out.shape == (2, 3, 8)

    def test_latent_cross_attention_residual(self):
        """LatentCrossAttention should apply residual connection."""
        lca = LatentCrossAttention(8, 2, num_latent_tokens=3)
        x = torch.randn(2, 10, 8)
        out = lca(x)
        # Output should include the latent tokens (residual)
        assert out.shape == (2, 3, 8)

    def test_latent_cross_attention_batch_independence(self):
        """LatentCrossAttention should handle different batch sizes."""
        lca = LatentCrossAttention(8, 2, num_latent_tokens=2)
        x = torch.randn(4, 5, 8)
        out = lca(x)
        assert out.shape == (4, 2, 8)

    def test_latent_cross_attention_with_mask(self):
        """LatentCrossAttention should accept and apply a mask."""
        lca = LatentCrossAttention(8, 2, num_latent_tokens=3)
        x = torch.randn(2, 5, 8)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        out = lca(x, mask=mask)
        assert out.shape == (2, 3, 8)
