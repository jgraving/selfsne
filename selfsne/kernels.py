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
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log Laplace kernel between two sets of points x and y, with a given scale.

    The log Laplace kernel is defined as:
    log(K(x, y)) = - ||x - y||_1 / scale

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The Laplace kernel matrix, of shape (batch_size,).

    """
    return (x - y).abs().sum(-1).div(scale).neg()


def pairwise_laplace(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log Laplace kernel between two sets of points x and y, with a given scale.

    The pairwise log Laplace kernel is defined as:
    log(K(x_i, y_j)) = - ||x_i - y_j||_1 / scale

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Laplace kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return torch.cdist(x, y.squeeze(), p=1).div(scale).neg().T


def cauchy(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log Cauchy kernel between two sets of points x and y, with a given scale.

    The log Cauchy kernel is defined as:
    log(K(x, y)) = - log(1 + ||x - y||_2^2 / scale^2)

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise Cauchy kernel matrix, of shape (batch_size,).

    """
    return (x - y).pow(2).sum(-1).div(scale ** 2).log1p().neg()


def pairwise_cauchy(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log Cauchy kernel between two sets of points x and y, with a given scale.

    The pairwise log Cauchy kernel is defined as:
    log(K(x_i, y_j)) = - log(1 + ||x_i - y_j||_2^2 / scale^2)

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise Cauchy kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return torch.cdist(x, y.squeeze(), p=2).div(scale).pow(2).log1p().neg().T


def inverse(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log inverse kernel between two sets of points x and y, with a given scale.

    The log inverse kernel is defined as:
    log(K(x, y)) = -log(eps + ||x - y||_2^2 / scale^2)

    where eps is a small constant to avoid taking the logarithm of zero.

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise inverse kernel matrix, of shape (batch_size,).

    """
    eps = 1e-5
    return -(x - y).pow(2).sum(-1).div(scale ** 2).add(eps).log()


def pairwise_inverse(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log inverse kernel between two sets of points x and y, with a given scale.

    The pairwise log inverse kernel is defined as:
    log(K(x_i, y_j)) = -log(eps + ||x_i - y_j||_2^2 / scale^2)

    where eps is a small constant to avoid taking the logarithm of zero.

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise inverse kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    eps = 1e-5
    return torch.cdist(x, y.squeeze(), p=2).div(scale).pow(2).add(eps).log().neg().T


def normal(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log normal kernel between two sets of points x and y, with a given scale.

    The log normal kernel is defined as:
    log(K(x, y)) = -1/2 * ||x - y||_2^2 / scale^2

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise normal kernel matrix, of shape (batch_size,).

    """
    return (x - y).div(scale).pow(2).sum(-1).div(2).neg()


def pairwise_normal(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log normal kernel between two sets of points x and y, with a given scale.

    The pairwise log normal kernel is defined as:
    log(K(x_i, y_j)) = -1/2 * ||x_i - y_j||_2^2 / scale^2

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise normal kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return torch.cdist(x, y.squeeze()).div(scale).pow(2).neg().T


def inner_product(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log inner product kernel between two sets of points x and y, with a given scale.

    The log inner product kernel is defined as:
    log(K(x, y)) = 1/scale * x.T @ y

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise inner product kernel matrix, of shape (batch_size,).

    """
    return (x * y).sum(-1).div(scale)


def pairwise_inner_product(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log inner product kernel between two sets of points x and y, with a given scale.

    The pairwise log inner product kernel is defined as:
    log(K(x_i, y_j)) = 1/scale * x_i.T @ y_j

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise inner product kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return (x @ y.squeeze().T).div(scale).T


def von_mises(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log von Mises-Fisher kernel between two sets of points x and y, with a given scale.

    The log von Mises-Fisher kernel is defined as:
    log(K(x, y)) = 1/scale * x_norm.T @ y_norm

    where x_norm = x / ||x||_2 and y_norm = y / ||y||_2 are the normalized versions of x and y.

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise von Mises-Fisher kernel matrix, of shape (batch_size,).

    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return inner_product(x_norm, y_norm, scale)


def pairwise_von_mises(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log von Mises-Fisher kernel between two sets of points x and y, with a given scale.

    The pairwise log von Mises-Fisher kernel is defined as:
    log(K(x_i, y_j)) = 1/scale * x_i_norm.T @ y_j_norm

    where x_i_norm = x_i / ||x_i||_2 and y_j_norm = y_j / ||y_j||_2 are the normalized versions of x_i and y_j.

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise von Mises-Fisher kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y.squeeze(), dim=-1)
    return pairwise_inner_product(x_norm, y_norm, scale).T


def wrapped_cauchy(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log wrapped Cauchy kernel between two sets of points x and y, with a given scale.

    The log wrapped Cauchy kernel is defined as:
    log(K(x, y)) = log(cosh(scale) - log(K_vmf(x, y, 1)))

    where K_vmf(x, y, 1) is the von Mises-Fisher kernel between x and y.

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise wrapped Cauchy kernel matrix, of shape (batch_size,).

    """
    return (np.cosh(scale) - von_mises(x, y, 1)).log().neg()


def pairwise_wrapped_cauchy(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log wrapped Cauchy kernel between two sets of points x and y, with a given scale.

    The pairwise log wrapped Cauchy kernel is defined as:
    log(K(x_i, y_j)) = log(cosh(scale) - log(K_vmf(x_i, y_j, 1)))

    where K_vmf(x_i, y_j, 1) is the von Mises-Fisher kernel between x_i and y_j.

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size_1, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size_2, num_points, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise wrapped Cauchy kernel matrix, of shape (batch_size_1, batch_size_2).

    """
    return (np.cosh(scale) - pairwise_von_mises(x, y, 1)).log().neg().T


def joint_product(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the log joint product kernel between two sets of points x and y, with a given scale.

    The log joint product kernel is defined as:
    log(K(x, y)) = log(Σ (softmax(x) * softmax(y))) / scale

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise joint product kernel matrix, of shape (batch_size,).

    """
    return (x.log_softmax(-1) + y.log_softmax(-1)).logsumexp(-1).div_(scale)


def pairwise_joint_product(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the pairwise log joint product kernel between two sets of points x and y, with a given scale.

    The log joint product kernel is defined as:
    log(K(x, y)) = log(Σ (softmax(x) * softmax(y))) / scale

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The pairwise joint product kernel matrix, of shape (batch_size, batch_size).

    """
    x_log_softmax = x.log_softmax(-1)
    y_log_softmax = y.log_softmax(-1)

    x_max = x_log_softmax.max(-1, keepdim=True)[0]
    y_max = y_log_softmax.max(-1, keepdim=True)[0]

    x_stable = x_log_softmax - x_max
    y_stable = y_log_softmax - y_max

    pairwise_matrix = torch.matmul(x_stable.exp(), y_stable.exp().T)
    log_pairwise_matrix = (
        torch.log(pairwise_matrix)
        + x_max.squeeze(-1)[:, None]
        + y_max.squeeze(-1)[None, :]
    )

    return log_pairwise_matrix.div_(scale).T


def bhattacharyya(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the Bhattacharyya kernel between two sets of points x and y, with a given scale.

    The Bhattacharyya kernel is defined as:
    log(K(x, y)) = log(Σ (softmax(x) * softmax(y))) / 2 * scale

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise Bhattacharyya kernel matrix, of shape (batch_size,).

    """
    return joint_product(x, y, scale * 2)


def pairwise_bhattacharyya(
    x: torch.Tensor, y: torch.Tensor, scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the Bhattacharyya kernel between two sets of points x and y, with a given scale.

    The Bhattacharyya kernel is defined as:
    log(K(x, y)) = log(Σ (softmax(x) * softmax(y))) / 2 * scale

    Args:
        x (torch.Tensor): The first set of points, of shape (batch_size, dim).
        y (torch.Tensor): The second set of points, of shape (batch_size, dim).
        scale (Union[float, torch.Tensor]): The scaling parameter.

    Returns:
        torch.Tensor: The row-wise Bhattacharyya kernel matrix, of shape (batch_size,).

    """
    return pairwise_joint_product(x, y, scale * 2)


KERNELS = {
    "normal": normal,
    "pairwise_normal": pairwise_normal,
    "student_t": cauchy,
    "pairwise_student_t": pairwise_cauchy,
    "cauchy": cauchy,
    "pairwise_cauchy": pairwise_cauchy,
    "inverse": inverse,
    "pairwise_inverse": pairwise_inverse,
    "laplace": laplace,
    "pairwise_laplace": pairwise_laplace,
    "von_mises": von_mises,
    "pairwise_von_mises": pairwise_von_mises,
    "wrapped_cauchy": wrapped_cauchy,
    "pairwise_wrapped_cauchy": pairwise_wrapped_cauchy,
    "inner_product": inner_product,
    "pairwise_inner_product": pairwise_inner_product,
    "joint_product": joint_product,
    "pairwise_joint_product": pairwise_joint_product,
    "bhattacharyya": bhattacharyya,
    "pairwise_bhattacharyya": pairwise_bhattacharyya,
}
