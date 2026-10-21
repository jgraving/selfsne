import math

import pytest
import torch

from selfsne.kernels import PAIRWISE_KERNELS, studentt, pairwise_studentt
from selfsne.losses import LikelihoodRatioEstimator


@pytest.mark.parametrize("scale", [0.3, 1.0, 2.0])
@pytest.mark.parametrize(
    "kernel", [studentt, pairwise_studentt, PAIRWISE_KERNELS["precise_student_t"]]
)
def test_student_t_scale_matches_known_distance(kernel, scale):
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    y = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    expected = -1.5 * math.log1p(25 / (2 * scale**2))
    torch.testing.assert_close(
        kernel(x, y, scale).squeeze(), torch.tensor(expected, dtype=torch.float64)
    )


@pytest.mark.parametrize("kernel", [studentt, pairwise_studentt])
def test_student_t_has_analytic_input_and_scale_gradients(kernel):
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    value = kernel(x, y, scale).sum()
    input_gradient, scale_gradient = torch.autograd.grad(value, (x, scale))
    denominator = 2 * scale.detach().square() + 25
    torch.testing.assert_close(input_gradient, 3 * y / denominator)
    torch.testing.assert_close(scale_gradient, 75 / (scale.detach() * denominator))


@pytest.mark.parametrize("explicit", [False, True])
def test_student_t_loss_applies_configured_kernel_scale(explicit):
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
    reference = (
        torch.tensor([[[3.0], [5.0]], [[2.0], [6.0]]], dtype=torch.float64)
        if explicit
        else None
    )
    loss = LikelihoodRatioEstimator(kernel="student_t", kernel_scale=2.0, baseline=0)
    actual = loss.logits(x, y, reference_embedding=reference)
    # In one dimension the Student-t kernel is the Cauchy kernel.
    cauchy = LikelihoodRatioEstimator(kernel="cauchy", kernel_scale=2.0, baseline=0)
    expected = cauchy.logits(x, y, reference_embedding=reference)
    for name in ("pos_logits", "neg_logits"):
        torch.testing.assert_close(actual[name], expected[name])
