import math

import pytest
import torch

from selfsne.data import correlated_gaussian_sampler


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("mutual_information", [0.0, 1e-8, 60.0])
def test_gaussian_samples_match_stable_analytic_conditional(
    dtype, dim, mutual_information
):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        samples = correlated_gaussian_sampler(7, mutual_information, dim=dim, seed=31)
        independent = torch.randn(
            7, 2 * dim, generator=torch.Generator().manual_seed(31)
        )
        x, noise = independent.split(dim, dim=1)
        # For isotropic paired Gaussians, I = -dim/2 * log(1-rho**2).
        rho = math.sqrt(-math.expm1(-2 * mutual_information / dim))
        conditional_scale = math.exp(-mutual_information / dim)
        expected_y = rho * x + conditional_scale * noise
        torch.testing.assert_close(samples[:, :dim], x, rtol=0, atol=0)
        torch.testing.assert_close(
            samples[:, dim:], expected_y, rtol=0, atol=8 * torch.finfo(dtype).eps
        )
    finally:
        torch.set_default_dtype(previous)
