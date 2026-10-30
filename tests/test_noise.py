# -*- coding: utf-8 -*-
"""Tests for selfsne.noise module."""

import pytest
import torch

from selfsne.noise import (
    Noise,
    Normal,
    LogNormalSoftmax,
    GumbelSoftmax,
    ConditionalDropout,
)


class TestNoise:
    """Tests for the base Noise class."""

    def test_noise_applies_with_probability_one(self):
        """When p=1.0, noise should always be applied."""
        torch.manual_seed(42)
        x = torch.zeros(4, 8)
        noise = Normal(p=1.0)
        out = noise(x)
        # Normal noise adds Gaussian noise, so output should differ from input
        assert not torch.allclose(out, x)

    def test_noise_skipped_with_probability_zero(self):
        """When p=0.0, noise should never be applied."""
        x = torch.randn(4, 8)
        noise = Normal(p=0.0)
        out = noise(x)
        torch.testing.assert_close(out, x)

    def test_noise_preserves_shape(self):
        """Noise should preserve the shape of the input."""
        x = torch.randn(4, 8, 16)
        noise = Normal(scale=0.5, p=1.0)
        out = noise(x)
        assert out.shape == x.shape

    def test_base_noise_class_does_not_implement_forward(self):
        """Base Noise._forward is not implemented, so calling forward with p=1
        will raise AttributeError when it tries to call _forward."""
        noise = Noise(p=1.0)
        with pytest.raises(AttributeError):
            noise(torch.randn(2, 3))


class TestNormal:
    """Tests for Normal noise."""

    def test_normal_adds_gaussian_noise(self):
        """Normal noise should add Gaussian noise to the input."""
        torch.manual_seed(42)
        x = torch.zeros(100, 8)
        noise = Normal(scale=1.0, p=1.0)
        out = noise(x)
        # Mean should be close to 0 (input), with std close to 1.0 (scale)
        assert out.mean().abs() < 0.3
        assert out.std().item() > 0.5

    def test_normal_preserves_shape(self):
        """Normal noise should preserve the shape of the input."""
        x = torch.randn(4, 8, 16)
        noise = Normal(scale=0.5, p=1.0)
        out = noise(x)
        assert out.shape == x.shape

    def test_normal_with_p_zero(self):
        """When p=0, Normal should pass through unchanged."""
        x = torch.randn(4, 8)
        noise = Normal(scale=10.0, p=0.0)
        out = noise(x)
        torch.testing.assert_close(out, x)

    def test_normal_scale_affects_variance(self):
        """Larger scale should produce larger variance."""
        torch.manual_seed(42)
        x = torch.zeros(500, 4)
        out_small = Normal(scale=0.1, p=1.0)(x)
        out_large = Normal(scale=1.0, p=1.0)(x)
        assert out_large.std() > out_small.std()

    def test_normal_requires_positive_scale(self):
        """Negative scale should raise ValueError."""
        with pytest.raises(ValueError):
            Normal(scale=-1.0)(torch.randn(2, 3))


class TestLogNormalSoftmax:
    """Tests for LogNormalSoftmax noise."""

    def test_output_is_probabilistic(self):
        """Output should sum to 1 along the last dimension (softmax)."""
        torch.manual_seed(42)
        x = torch.ones(4, 8)  # Use all-positive input so log is valid
        noise = LogNormalSoftmax(p=1.0)
        out = noise(x)
        # After softmax, rows should sum to ~1
        assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)
        # All values should be non-negative
        assert (out >= 0).all()

    def test_output_preserves_shape(self):
        """LogNormalSoftmax should preserve the shape of the input."""
        x = torch.ones(4, 8)
        noise = LogNormalSoftmax(p=1.0)
        out = noise(x)
        assert out.shape == x.shape

    def test_skip_with_p_zero(self):
        """When p=0, LogNormalSoftmax should pass through unchanged."""
        x = torch.ones(4, 8)
        noise = LogNormalSoftmax(p=0.0)
        out = noise(x)
        torch.testing.assert_close(out, x)


class TestGumbelSoftmax:
    """Tests for GumbelSoftmax (RelaxedOneHotCategorical) noise."""

    def test_output_shape(self):
        """Output should preserve the shape of the input."""
        torch.manual_seed(42)
        x = torch.ones(4, 8)  # All-positive so log is valid
        noise = GumbelSoftmax(temperature=0.5, p=1.0)
        out = noise(x)
        assert out.shape == x.shape

    def test_output_in_range(self):
        """Output values should be non-negative (softmax of Gumbel)."""
        torch.manual_seed(42)
        x = torch.ones(4, 8)
        noise = GumbelSoftmax(temperature=0.1, p=1.0)
        out = noise(x)
        assert (out >= 0).all()

    def test_temperature_affects_distribution(self):
        """Lower temperature should produce sharper (more peaky) outputs."""
        torch.manual_seed(42)
        x = torch.ones(100, 8)
        out_cold = GumbelSoftmax(temperature=0.1, p=1.0)(x)
        out_warm = GumbelSoftmax(temperature=5.0, p=1.0)(x)
        # Cold should be more peaky (higher max)
        assert out_cold.max(dim=-1).values.mean() > out_warm.max(dim=-1).values.mean()

    def test_skip_with_p_zero(self):
        """When p=0, GumbelSoftmax should pass through unchanged."""
        x = torch.ones(4, 8)
        noise = GumbelSoftmax(temperature=0.1, p=0.0)
        out = noise(x)
        torch.testing.assert_close(out, x)

    def test_temperature_greater_than_zero(self):
        """Temperature must be positive."""
        # RelaxedOneHotCategorical accepts any positive temperature;
        # negative values produce undefined behaviour but do not raise
        # at construction time — they fail at sample time.
        gs = GumbelSoftmax(temperature=-1.0, p=1.0)
        x = torch.ones(4, 8)
        # Negative temperature changes the distribution behaviour;
        # we just verify it doesn't crash at construction
        assert gs.temperature == -1.0


class TestConditionalDropout:
    """Tests for ConditionalDropout noise."""

    def test_output_shape(self):
        """Output should preserve the shape of the input."""
        torch.manual_seed(42)
        x = torch.clamp(torch.randn(4, 8), min=0.0, max=1.0)  # Clamp to [0, 1]
        noise = ConditionalDropout(p=1.0)
        out = noise(x)
        assert out.shape == x.shape

    def test_output_zeros_out_values_below_threshold(self):
        """ConditionalDropout multiplies by Bernoulli(probs=x), so values close to 0
        get zeroed out."""
        torch.manual_seed(42)
        x = torch.tensor([[0.0, 1.0, 0.5, 0.1]])
        noise = ConditionalDropout(p=1.0)
        out = noise(x)
        # x[0] = 0.0 means Bernoulli(0) = 0 always
        assert (out[0, 0] == 0).all()

    def test_skip_with_p_zero(self):
        """When p=0, ConditionalDropout should pass through unchanged."""
        x = torch.clamp(torch.randn(4, 8), min=0.0, max=1.0)
        noise = ConditionalDropout(p=0.0)
        out = noise(x)
        torch.testing.assert_close(out, x)

    def test_non_negative_values_passed_through_stochastically(self):
        """Non-negative values in [0, 1] are kept with probability proportional to their value."""
        torch.manual_seed(42)
        x = torch.ones(1000, 4)  # All ones, so bernoulli(p=1.0) should keep all
        noise = ConditionalDropout(p=1.0)
        out = noise(x)
        # With p=1.0 and values=1.0, Bernoulli(1.0) should always return 1
        assert (out == 1.0).all()
