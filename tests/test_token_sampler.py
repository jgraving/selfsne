# -*- coding: utf-8 -*-
"""Tests for selfsne.nn.sampling module (in addition to test_sequences.py)."""

import pytest
import torch
import warnings

from selfsne.nn.sampling import TokenSampler


class TestTokenSampler:
    """Tests for TokenSampler module."""

    def test_eval_mode_returns_unchanged(self):
        """TokenSampler should return input unchanged in eval mode."""
        x = torch.randn(2, 10, 8)
        sampler = TokenSampler(p=0.5)
        sampler.eval()
        out = sampler(x)
        torch.testing.assert_close(out, x)

    def test_training_mode_reduces_tokens(self):
        """TokenSampler should reduce tokens in training mode."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 8)
        sampler = TokenSampler(p=0.3)  # Keep 30% = 3 tokens
        sampler.train()
        out = sampler(x)
        assert out.shape == (2, 3, 8)

    def test_num_tokens_exact_count(self):
        """TokenSampler with num_tokens should keep exactly that many tokens."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 8)
        sampler = TokenSampler(num_tokens=5)
        sampler.train()
        out = sampler(x)
        assert out.shape == (2, 5, 8)

    def test_num_tokens_capped_at_sequence_length(self):
        """TokenSampler should not keep more tokens than available."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, 8)
        sampler = TokenSampler(num_tokens=100)  # More than available
        sampler.train()
        out = sampler(x)
        assert out.shape == (2, 3, 8)

    def test_fractional_sampling_keeps_at_least_one(self):
        """TokenSampler should keep at least 1 token even with very small p."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, 8)
        sampler = TokenSampler(p=0.01)
        sampler.train()
        out = sampler(x)
        assert out.shape[1] >= 1

    def test_no_warning_with_p_specified(self):
        """TokenSampler should not warn when p is specified."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TokenSampler(p=0.5)

    def test_no_warning_with_num_tokens_specified(self):
        """TokenSampler should not warn when num_tokens is specified."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TokenSampler(num_tokens=5)

    def test_warns_without_p_or_num_tokens(self):
        """TokenSampler should warn when neither p nor num_tokens is specified."""
        with pytest.warns(UserWarning, match="Neither"):
            TokenSampler()

    def test_rejects_both_p_and_num_tokens(self):
        """TokenSampler should reject specifying both p and num_tokens."""
        with pytest.raises(ValueError, match="not both"):
            TokenSampler(p=0.5, num_tokens=5)

    def test_rejects_invalid_p(self):
        """TokenSampler should reject p <= 0 or p > 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenSampler(p=0.0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenSampler(p=-0.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenSampler(p=1.5)

    def test_rejects_invalid_num_tokens(self):
        """TokenSampler should reject non-positive or non-integer num_tokens."""
        with pytest.raises(ValueError, match="positive"):
            TokenSampler(num_tokens=0)
        with pytest.raises(ValueError, match="positive"):
            TokenSampler(num_tokens=-1)
        with pytest.raises(ValueError, match="positive"):
            TokenSampler(num_tokens=0.5)
        with pytest.raises(ValueError, match="positive"):
            TokenSampler(num_tokens=True)  # bool is subclass of int

    def test_preserves_features(self):
        """TokenSampler should preserve the feature dimension."""
        x = torch.randn(2, 10, 16)
        sampler = TokenSampler(p=0.5)
        sampler.train()
        out = sampler(x)
        assert out.shape[2] == 16

    def test_different_random_samples_each_call(self):
        """TokenSampler should produce different samples on each call in training mode."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 8)  # Use random input, not constant
        sampler = TokenSampler(p=0.5)
        sampler.train()
        out1 = sampler(x.clone())
        out2 = sampler(x.clone())
        # With p=0.5, different random subsets should be selected
        # The outputs will differ because different tokens are selected
        assert not torch.allclose(out1, out2)
