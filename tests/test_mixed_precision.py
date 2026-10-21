import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator


@pytest.mark.parametrize(
    "embeddings,logits", [(True, False), (False, True), (True, True)]
)
def test_bfloat16_autocast_predictions_are_float32_numpy(embeddings, logits):
    model = SelfSNE(
        loss=LikelihoodRatioEstimator(divergence="bce", baseline="embedding"),
        target_encoder=nn.Linear(4, 2),
        baseline_head=nn.Linear(2, 2),
        return_embeddings=embeddings,
        return_logits=logits,
    ).eval()
    batch = {
        "context": torch.randn(4, 4),
        "target": torch.randn(4, 4),
        "reference": torch.randn(4, 3, 4),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model.predict_step(batch, 0)
    assert result
    for value in result.values():
        assert value.dtype == np.float32
        assert np.isfinite(value).all()


@pytest.mark.parametrize(
    "kernel",
    [
        "normal",
        "cauchy",
        "student_t",
        "inverse",
        "laplace",
        "inner_product",
        "von_mises",
        "wrapped_cauchy",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float16,
            marks=pytest.mark.skipif(
                sys.platform.startswith("linux") and not Path("/proc/cpuinfo").exists(),
                reason="PyTorch CPU float16 conversion requires /proc/cpuinfo on Linux.",
            ),
        ),
        torch.bfloat16,
    ],
)
def test_low_precision_kernel_scores_and_gradients_match_float32(kernel, dtype):
    context = torch.tensor(
        [[300.0, -200.0], [-250.0, 300.0]], dtype=dtype, requires_grad=True
    )
    target = torch.tensor(
        [[-300.0, 200.0], [250.0, -300.0]], dtype=dtype, requires_grad=True
    )
    reference = (
        target.detach().float()[None].expand(2, -1, -1).to(dtype).requires_grad_()
    )
    loss = LikelihoodRatioEstimator(kernel=kernel, baseline=0, divergence="bce")
    actual = loss(context, target, reference)["loss"]
    x = context.detach().float().requires_grad_()
    y = target.detach().float().requires_grad_()
    refs = reference.detach().float().requires_grad_()
    expected = loss(x, y, refs)["loss"]
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for low, high in [(context, x), (target, y), (reference, refs)]:
        assert torch.isfinite(low.grad).all()
        torch.testing.assert_close(low.grad, high.grad.to(dtype))


def test_full_precision_prediction_dtype_is_preserved():
    model = (
        SelfSNE(loss=LikelihoodRatioEstimator(), target_encoder=nn.Linear(4, 2))
        .double()
        .eval()
    )
    result = model.predict_step(torch.randn(4, 4, dtype=torch.float64), 0)
    assert result["target_embedding"].dtype == np.float64


def test_custom_kernel_retains_its_precision_contract():
    class CustomKernel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 2)

        def forward(self, x, y, scale):
            return (self.projection(x) * y).sum(-1)

    loss = LikelihoodRatioEstimator(
        kernel=CustomKernel().bfloat16(), baseline=0, divergence="bce"
    )
    x = torch.randn(4, 2, dtype=torch.bfloat16, requires_grad=True)
    value = loss(x, x)["loss"]
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(x.grad).all()
