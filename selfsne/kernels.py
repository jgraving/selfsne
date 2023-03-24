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
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log Laplace kernel between two sets of points x1 and x2, with a given scale.

    The log Laplace kernel is defined as:
    log(K(x1, x2)) = - ||x1 - x2||_1 / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The Laplace kernel matrix, of shape (batch_size,).

    """
    return (x1 - x2).abs().sum(-1).div(scale).neg()


def pairwise_laplace(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log Laplace kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log Laplace kernel is defined as:
    log(K(x1_i, x2_j)) = - ||x1_i - x2_j||_1 / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Laplace kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return torch.cdist(x1, x2, p=1).div(scale).neg()


def cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The log Cauchy kernel is defined as:
    log(K(x1, x2)) = - log(1 + ||x1 - x2||_2^2 / scale^2)

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise Cauchy kernel matrix, of shape (batch_size,).

    """
    return (x1 - x2).pow(2).sum(-1).div(scale ** 2).log1p().neg()


def pairwise_cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log Cauchy kernel is defined as:
    log(K(x1_i, x2_j)) = - log(1 + ||x1_i - x2_j||_2^2 / scale^2)

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Cauchy kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return torch.cdist(x1, x2, p=2).div(scale).pow(2).log1p().neg()


def inverse(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log inverse kernel between two sets of points x1 and x2, with a given scale.

    The log inverse kernel is defined as:
    log(K(x1, x2)) = -log(eps + ||x1 - x2||_2^2 / scale^2)

    where eps is a small constant to avoid taking the logarithm of zero.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise inverse kernel matrix, of shape (batch_size,).

    """
    eps = 1e-5
    return -(x1 - x2).pow(2).sum(-1).div(scale ** 2).add(eps).log()


def pairwise_inverse(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log inverse kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log inverse kernel is defined as:
    log(K(x1_i, x2_j)) = -log(eps + ||x1_i - x2_j||_2^2 / scale^2)

    where eps is a small constant to avoid taking the logarithm of zero.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise inverse kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    eps = 1e-5
    return torch.cdist(x1, x2, p=2).div(scale).pow(2).add(eps).log().neg()


def normal(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log normal kernel between two sets of points x1 and x2, with a given scale.

    The log normal kernel is defined as:
    log(K(x1, x2)) = -1/2 * ||x1 - x2||_2^2 / scale^2

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise normal kernel matrix, of shape (batch_size,).

    """
    return (x1 - x2).div(scale).pow(2).sum(-1).div(2).neg()


def pairwise_normal(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log normal kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log normal kernel is defined as:
    log(K(x1_i, x2_j)) = -1/2 * ||x1_i - x2_j||_2^2 / scale^2

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise normal kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return torch.cdist(x1, x2).div(scale).pow(2).neg()


def inner_product(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the inner product kernel between two sets of points x1 and x2, with a given scale.

    The log inner product kernel is defined as:
    log(K(x1, x2)) = 1/scale * x1.T @ x2

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise inner product kernel matrix, of shape (batch_size,).

    """
    return (x1 * x2).sum(-1).div(scale)


def pairwise_inner_product(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise inner product kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log inner product kernel is defined as:
    log(K(x1_i, x2_j)) = 1/scale * x1_i.T @ x2_j

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise inner product kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return (x2 @ x1.T).div(scale)


def von_mises(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log von Mises-Fisher kernel between two sets of points x1 and x2, with a given scale.

    The log von Mises-Fisher kernel is defined as:
    log(K(x1, x2)) = 1/scale * x1_norm.T @ x2_norm

    where x1_norm = x1 / ||x1||_2 and x2_norm = x2 / ||x2||_2 are the normalized versions of x1 and x2.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise von Mises-Fisher kernel matrix, of shape (batch_size,).

    """
    x1_norm = x1.norm(dim=-1)
    x2_norm = x2.norm(dim=-1)
    return inner_product(x1, x2, scale) / (x_norm * y_norm).clamp(min=eps)


def pairwise_von_mises(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the pairwise log von Mises-Fisher kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log von Mises-Fisher kernel is defined as:
    log(K(x1_i, x2_j)) = 1/scale * x1_i_norm.T @ x2_j_norm

    where x1_i_norm = x1_i / ||x1_i||_2 and x2_j_norm = x2_j / ||x2_j||_2 are the normalized versions of x1_i and x2_j.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise von Mises-Fisher kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    x1 = x1
    x1_norm = x1.norm(dim=-1, keepdim=True)
    x2_norm = x2.norm(dim=-1, keepdim=True)
    return pairwise_inner_product(x1_norm, x2_norm, scale) / (x_norm * y_norm.T).clamp(
        min=eps
    )


def wrapped_cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log wrapped Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The log wrapped Cauchy kernel is defined as:
    log(K(x1, x2)) = log(cosh(scale) - log(K_vmf(x1, x2, 1)))

    where K_vmf(x1, x2, 1) is the von Mises-Fisher kernel between x1 and x2.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise wrapped Cauchy kernel matrix, of shape (batch_size,).

    """
    return (torch.cosh(scale) - von_mises(x1, x2, 1)).log().neg()


def pairwise_wrapped_cauchy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log wrapped Cauchy kernel between two sets of points x1 and x2, with a given scale.

    The pairwise log wrapped Cauchy kernel is defined as:
    log(K(x1_i, x2_j)) = log(cosh(scale) - log(K_vmf(x1_i, x2_j, 1)))

    where K_vmf(x1_i, x2_j, 1) is the von Mises-Fisher kernel between x1_i and x2_j.

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise wrapped Cauchy kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return (np.cosh(scale) - pairwise_von_mises(x1, x2, 1)).log().neg()


def joint_product(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log joint product kernel between two sets of points x1 and x2, with a given scale.

    The log joint product kernel is defined as:
    log(K(x1, x2)) = log(Σ (softmax(x1) * softmax(x2))) / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise joint product kernel matrix, of shape (batch_size,).

    """
    return (x1.log_softmax(-1) + x2.log_softmax(-1)).logsumexp(-1).div_(scale)


def pairwise_joint_product(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log joint product kernel between two sets of points x1 and x2, with a given scale.


    The log joint product kernel is defined as:
    log(K(x1_i, x2_j)) = log(Σ (softmax(x1_i) * softmax(x2_j))) / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise joint product kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    x1_log_softmax = x1.log_softmax(-1)
    x2_log_softmax = x2.log_softmax(-1)

    x1_max = x1_log_softmax.max(-1, keepdim=True)[0]
    x2_max = x2_log_softmax.max(-1, keepdim=True)[0]

    x1_stable = x1_log_softmax - x1_max
    x2_stable = x2_log_softmax - x2_max

    pairwise_matrix = torch.matmul(x1_stable.exp(), x2_stable.exp().T)
    log_pairwise_matrix = torch.log(pairwise_matrix) + x1_max + x2_max.T

    return log_pairwise_matrix / scale


def cross_entropy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the cross entropy between two sets of points x1 and x2.

    The cross entropy is defined as:
    H(x1_i, x2_i) = - Σ(softmax(x1_i) * log_softmax(x2_i)) / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise cross entropy matrix, of shape (batch_size,).
    """
    return (x1.softmax(-1) * x2.log_softmax(-1)).sum(-1) / scale


def pairwise_cross_entropy(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise cross entropy between two sets of points x1 and x2.

    The pairwise cross entropy is defined as:
    H(x1_i, x2_j) = - Σ(softmax(x1_i) * log_softmax(x2_j)) / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise cross entropy matrix, of shape (batch_size_1, batch_size_2).
    """
    return (x1.softmax(-1) @ x2.log_softmax(-1).T) / scale


def kl_divergence(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence between two sets of points x1 and x2.

    The Kullback-Leibler divergence is defined as:
    KL(x1_i || x2_i) = Σ(softmax(x1_i) * (log_softmax(x1_i) - log_softmax(x2_i))) / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise divergence matrix, of shape (batch_size,).
    """
    log_p1 = x1.log_softmax(-1)
    log_p2 = x2.log_softmax(-1)
    p1 = log_p1.exp()
    return ((p1 * (log_p1 - log_p2)).sum(-1)) / scale


def pairwise_kl_divergence(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise Kullback-Leibler divergence between two sets of points x1 and x2.

    The pairwise Kullback-Leibler divergence is defined as:
    KL(x1_i || x2_j) = Σ(softmax(x1_i) * (log_softmax(x1_i) - log_softmax(x2_j))) / scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise divergence matrix, of shape (batch_size_1, batch_size_2).
    """
    log_p1 = x1.log_softmax(-1)
    log_p2 = x2.log_softmax(-1)
    p1 = log_p1.exp()
    return (p1 @ (log_p1 - log_p2).T) / scale


def bhattacharyya(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the Bhattacharyya kernel between two sets of points x1 and x2, with a given scale.

    The Bhattacharyya kernel is defined as:
    log(K(x1, x2)) = log(Σ (softmax(x1) * softmax(x2))) / 2 * scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise Bhattacharyya kernel matrix, of shape (batch_size,).

    """
    return joint_product(x1, x2, scale * 2)


def pairwise_bhattacharyya(
    x1: torch.Tensor, x2: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise Bhattacharyya kernel between two sets of points x1 and x2, with a given scale.

    The Bhattacharyya kernel is defined as:
    log(K(x1_i, x2_j)) = log(Σ (softmax(x1_i) * softmax(x2_j))) / 2 * scale

    Args:
        x1 (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        x2 (torch.Tensor): The second set of points, of shape (batch_size_2, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Bhattacharyya kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return pairwise_joint_product(x1, x2, scale * 2)


ROWWISE_KERNELS = {
    "euclidean": normal,
    "normal": normal,
    "student_t": cauchy,
    "cauchy": cauchy,
    "inverse": inverse,
    "laplace": laplace,
    "von_mises": von_mises,
    "cosine": von_mises,
    "wrapped_cauchy": wrapped_cauchy,
    "inner_product": inner_product,
    "bilinear": inner_product,
    "joint_product": joint_product,
    "cross_entropy": cross_entropy,
    "softmax": cross_entropy,
    "kl_divergence": kl_divergence,
    "kld": kl_divergence,
    "kl_div": kl_divergence,
    "bhattacharyya": bhattacharyya,
    "hellinger": bhattacharyya,
}

PAIRWISE_KERNELS = {
    "euclidean": pairwise_normal,
    "normal": pairwise_normal,
    "student_t": pairwise_cauchy,
    "cauchy": pairwise_cauchy,
    "inverse": pairwise_inverse,
    "laplace": pairwise_laplace,
    "von_mises": pairwise_von_mises,
    "cosine": pairwise_von_mises,
    "wrapped_cauchy": pairwise_wrapped_cauchy,
    "inner_product": pairwise_inner_product,
    "bilinear": pairwise_inner_product,
    "joint_product": pairwise_joint_product,
    "cross_entropy": pairwise_cross_entropy,
    "softmax": pairwise_cross_entropy,
    "kl_divergence": pairwise_kl_divergence,
    "kld": pairwise_kl_divergence,
    "kl_div": pairwise_kl_divergence,
    "bhattacharyya": pairwise_bhattacharyya,
    "hellinger": pairwise_bhattacharyya,
}
