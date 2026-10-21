"""CUDA checks skip on CPU CI, but are mandatory when explicitly requested."""

import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA hardware is available.",
    )


def pytest_configure(config):
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.getoption("--require-cuda"):
        if not torch.cuda.is_available():
            raise pytest.UsageError(
                "CUDA validation requires a CUDA-enabled PyTorch build and an accessible NVIDIA GPU; CPU fallback is disabled."
            )


def pytest_collection_modifyitems(config, items):
    cuda_items = [item for item in items if item.get_closest_marker("cuda")]
    if config.getoption("--require-cuda") and not cuda_items:
        raise pytest.UsageError(
            "No CUDA tests selected; CUDA validation cannot pass without executing them."
        )
    if not torch.cuda.is_available():
        for item in cuda_items:
            item.add_marker(pytest.mark.skip(reason="CUDA hardware is not available."))
