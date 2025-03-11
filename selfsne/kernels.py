# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import Module
import torch.nn.functional as F
import numpy as np

from typing import Union


def laplace(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the log Laplace kernel between two sets of points x1 and x2, with a given scale.

    The log Laplace kernel is defined as:
    K(x1, x2) = - ||x1 - x2||_1 / (scale * sqrt(dim))

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The Laplace kernel matrix, of shape (batch_size,).

    """
    dim_scale = scale * np.sqrt(x1.shape[-1])
    return (x1 - x2).abs().sum(-1).div(dim_scale).neg()


def pairwise_laplace(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the pairwise log Laplace kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log Laplace kernel is defined as:
    K(x1_i, x2_j) = - ||x1_i - x2_j||_1 / (scale * sqrt(dim))

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Laplace kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    dim_scale = scale * np.sqrt(x1.shape[-1])
    return torch.cdist(x1, x2, p=1).div(dim_scale).neg()


def studentt(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Computes the log Student's t kernel between two sets of points x1 and x2.

    The log Student's t kernel is defined as:
    K(x1, x2) = - log(1 + ||x1 - x2||_2^2 / dim) * (dim + 1) / 2

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).

    Returns:
        torch.Tensor: The row-wise Student's t kernel matrix, of shape (batch_size,).
    """
    dim = x1.shape[-1]  # df = dim
    return (x1 - x2).pow(2).sum(-1).div(dim).log1p().neg() * (dim + 1) / 2


def pairwise_studentt(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Computes the pairwise log Student's t kernel between two sets of points x1 and x2.

    The pairwise log Student's t kernel is defined as:
    K(x1_i, x2_j) = - log(1 + ||x1_i - x2_j||_2^2 / dim) * (dim + 1) / 2

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).

    Returns:
        torch.Tensor: The pairwise Student's t kernel matrix, of shape (batch_size_1, batch_size_2).
    """
    dim = x1.shape[-1]  # df = dim
    return torch.cdist(x1, x2, p=2).pow(2).div(dim).log1p().neg() * (dim + 1) / 2


def cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the log Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The log Cauchy kernel is defined as:
    K(x1, x2) = - log(1 + ||x1 - x2||_2^2 / (scale^2 * dim))

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise Cauchy kernel matrix, of shape (batch_size,).

    """
    dim_scale = (scale ** 2) * x1.shape[-1]
    return (x1 - x2).pow(2).sum(-1).div(dim_scale).log1p().neg()


def pairwise_cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the pairwise log Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log Cauchy kernel is defined as:
    K(x1_i, x2_j) = - log(1 + ||x1_i - x2_j||_2^2 / (scale^2 * dim))

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Cauchy kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    dim_scale = (scale ** 2) * x1.shape[-1]
    return torch.cdist(x1, x2, p=2).pow(2).div(dim_scale).log1p().neg()


def inverse(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the log inverse kernel between two sets of points x1 and x2, with a given scale.

    The log inverse kernel is defined as:
    K(x1, x2) = -log(eps + ||x1 - x2||_2^2 / (scale^2 * dim))

    where eps is a small constant to avoid taking the logarithm of zero.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise inverse kernel matrix, of shape (batch_size,).

    """
    dim_scale = (scale ** 2) * x1.shape[-1]
    eps = torch.finfo(x1.dtype).eps
    return -(x1 - x2).pow(2).sum(-1).div(dim_scale).clamp(min=eps).log()


def pairwise_inverse(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the pairwise log inverse kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log inverse kernel is defined as:
    K(x1_i, x2_j) = -log(eps + ||x1_i - x2_j||_2^2 / (scale^2 * dim))

    where eps is a small constant to avoid taking the logarithm of zero.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise inverse kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    dim_scale = (scale ** 2) * x1.shape[-1]
    eps = torch.finfo(x1.dtype).eps
    return torch.cdist(x1, x2, p=2).pow(2).div(dim_scale).clamp(min=eps).log().neg()


def normal(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the log normal kernel between two sets of points x1 and x2, with a given scale.

    The log normal kernel is defined as:
    K(x1, x2) = -1/2 * ||x1 - x2||_2^2 / (scale^2 * dim)

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise normal kernel matrix, of shape (batch_size,).

    """
    dim_scale = (scale ** 2) * x1.shape[-1]
    return (x1 - x2).pow(2).sum(-1).div(dim_scale).div(2).neg()


def pairwise_normal(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the pairwise log normal kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log normal kernel is defined as:
    K(x1_i, x2_j) = -1/2 * ||x1_i - x2_j||_2^2 / (scale^2 * dim)

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise normal kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    dim_scale = (scale ** 2) * x1.shape[-1]
    return torch.cdist(x1, x2, p=2).pow(2).div(dim_scale).div(2).neg()


def inner_product(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the inner product kernel between two sets of points x1 and x2, with a given scale.

    The log inner product kernel is defined as:
    K(x1, x2) = 1/scale * x1.T @ x2

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise inner product kernel matrix, of shape (batch_size,).

    """
    dim_scale = scale * np.sqrt(x1.shape[-1])
    return (x1 * x2).sum(-1).div(dim_scale)


def pairwise_inner_product(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the pairwise inner product kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log inner product kernel is defined as:
    K(x1_i, x2_j) = 1/scale * x1_i.T @ x2_j

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise inner product kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    dim_scale = scale * np.sqrt(x1.shape[-1])
    return (x1 @ x2.T).div(dim_scale)


def von_mises(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Computes the log von Mises-Fisher kernel between two sets of points x1 and x2, with a given scale.

    The log von Mises-Fisher kernel is defined as:
    K(x1, x2) = 1/scale * cosine_similarity(x1, x2)

    where cosine_similarity(x1, x2) is the cosine similarity between x1 and x2.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise von Mises-Fisher kernel matrix, of shape (batch_size,).
    """
    eps = torch.finfo(x1.dtype).eps
    x1_norm = x1.norm(dim=-1)
    x2_norm = x2.norm(dim=-1)
    return (x1 * x2).sum(-1) / (scale * x1_norm * x2_norm).clamp(min=eps)


def pairwise_von_mises(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Computes the pairwise log von Mises-Fisher kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log von Mises-Fisher kernel is defined as:
    K(x1_i, x2_j) = 1/scale * cosine_similarity(x1_i, x2_j)

    where cosine_similarity(x1_i, x2_j) is the cosine similarity between x1_i and x2_j.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise von Mises-Fisher kernel matrix, of shape (batch_size_1, batch_size_2).
    """
    x1_norm = F.normalize(x1, p=2, dim=-1, eps=1e-8)
    x2_norm = F.normalize(x2, p=2, dim=-1, eps=1e-8)

    return (x1_norm @ x2_norm.T).div(scale)


def wrapped_cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the log wrapped Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The log wrapped Cauchy kernel is defined as:
    K(x1, x2) = log(cosh(scale) - log(K_vmf(x1, x2, 1)))

    where K_vmf(x1, x2, 1) is the von Mises-Fisher kernel between x1 and x2.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise wrapped Cauchy kernel matrix, of shape (batch_size,).

    """
    return (torch.sinh(scale)).log() - (torch.cosh(scale) - von_mises(x1, x2, 1)).log()


def pairwise_wrapped_cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """
    Computes the pairwise log wrapped Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log wrapped Cauchy kernel is defined as:
    K(x1_i, x2_j) = log(cosh(scale) - log(K_vmf(x1_i, x2_j, 1)))

    where K_vmf(x1_i, x2_j, 1) is the von Mises-Fisher kernel between x1_i and x2_j.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise wrapped Cauchy kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return (np.cosh(scale) - pairwise_von_mises(x1, x2, 1)).log().neg()


def pairwise_kernel(func):
    """
    Decorator that converts a rowwise kernel function into a pairwise kernel function by unsqueezing x1.

    Args:
        func (Callable): The rowwise kernel function.

    Returns:
        Callable: The pairwise kernel function.

    """

    def decorator(
        x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """
        Computes the pairwise kernel by unsqueezing x1 and calling the rowwise kernel function.

        Args:
            x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
            x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
            scale (Union[float, torch.Tensor]): The scaling parameter.

        Returns:
            torch.Tensor: The pairwise kernel matrix, of shape (batch_size_1, batch_size_2).

        """
        return func(x1.unsqueeze(1), x2, scale)

    return decorator


ROWWISE_KERNELS = {
    "euclidean": normal,
    "normal": normal,
    "student_t": studentt,
    "cauchy": cauchy,
    "inverse": inverse,
    "laplace": laplace,
    "von_mises": von_mises,
    "cosine": von_mises,
    "wrapped_cauchy": wrapped_cauchy,
    "inner_product": inner_product,
}

PAIRWISE_KERNELS = {
    "euclidean": pairwise_normal,
    "normal": pairwise_normal,
    "student_t": pairwise_studentt,
    "cauchy": pairwise_cauchy,
    "inverse": pairwise_inverse,
    "laplace": pairwise_laplace,
    "von_mises": pairwise_von_mises,
    "cosine": pairwise_von_mises,
    "wrapped_cauchy": pairwise_wrapped_cauchy,
    "inner_product": pairwise_inner_product,
    "bilinear": pairwise_inner_product,
}

# Define and add the "precise" versions of the pairwise kernels using the decorator
for kernel_name, kernel_func in ROWWISE_KERNELS.items():
    precise_kernel_name = f"precise_{kernel_name}"
    PAIRWISE_KERNELS[precise_kernel_name] = pairwise_kernel(kernel_func)
