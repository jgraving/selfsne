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

import pytorch_lightning as pl
import numpy as np

import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Optional, Tuple, List, Dict, Union


def inverse_softplus(x):
    """
    Computes the inverse of the softplus function for a given output value.

    Args:
        x (Tensor): A tensor containing the output value(s) of the softplus function.

    Returns:
        Tensor: A tensor containing the inverse of the softplus function for the input `x`.

    Notes:
        The softplus function is defined as `log(1 + exp(x))`, and the inverse softplus
        function is defined as `log(exp(x) - 1)`.
    """
    return x.expm1().log()


def stop_gradient(x: torch.Tensor) -> torch.Tensor:
    r"""
    Creates a new tensor from the input tensor with detached gradient.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        A new tensor with the same data as the input tensor, but detached from the computation graph.

    Examples::
        >>> x = torch.tensor([1, 2, 3], requires_grad=True)
        >>> y = stop_gradient(x)
        >>> y.requires_grad
        False
    """
    return x.clone().detach()


def set_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    r"""
    Set the gradient computation flag for all parameters of a given module.

    Args:
        module (torch.nn.Module): The module for which to set the gradient computation flag.
        requires_grad (bool): Whether to enable gradient computation (True) or disable it (False) for all parameters of the module.

    Returns:
        None

    Examples::
        >>> m = torch.nn.Linear(3, 4)
        >>> set_grad(m, False)
        >>> m.weight.requires_grad
        False
    """
    for param in module.parameters():
        param.requires_grad = requires_grad


def random_sample_columns(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    r"""
    Randomly selects a fixed number of columns from a 2D tensor without replacement.

    Args:
        x (torch.Tensor): The input tensor of shape :math:`(N, D)`.
        num_samples (int): The number of columns to randomly select.

    Returns:
        A tensor of shape :math:`(N, num\_samples)` containing the randomly selected columns.

    Examples::
        >>> x = torch.randn(10, 20)
        >>> random_sample_columns(x, 5).shape
        torch.Size([10, 5])
    """
    assert (
        num_samples <= x.shape[1]
    ), "num_samples must be less than or equal to the number of columns"
    weights = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    jdx = torch.multinomial(weights, num_samples, replacement=False)
    idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, num_samples)
    return x[idx, jdx]


def disable_grad(module: torch.nn.Module) -> None:
    r"""
    Disables gradient computation for all parameters of a given module.

    Args:
        module (torch.nn.Module): The module for which to disable gradient computation.

    Returns:
        None

    Examples::
        >>> m = torch.nn.Linear(3, 4)
        >>> disable_grad(m)
        >>> m.weight.requires_grad
        False
    """
    module.eval()
    set_grad(module, False)


def enable_grad(module: torch.nn.Module) -> None:
    r"""
    Enables gradient computation for all parameters of a given module.

    Args:
        module (torch.nn.Module): The module for which to enable gradient computation.

    Returns:
        None

    Examples::
        >>> m = torch.nn.Linear(3, 4)
        >>> enable_grad(m)
        >>> m.weight.requires_grad
        True
    """
    module.train()
    set_grad(module, True)


def reassign_labels(labels: np.ndarray, label_dict: Dict[int, int]) -> np.ndarray:
    r"""
    Reassigns labels in an array based on a dictionary.

    Args:
        labels (np.ndarray): A 1D numpy array of labels to be reassigned.
        label_dict (dict): A dictionary where the keys are the old labels and the values are the new labels.

    Returns:
        A new 1D numpy array with reassigned labels.

    Examples::
        >>> labels = np.array([0, 1, 2, 1, 0])
        >>> label_dict = {0: 1, 1: 2, 2: 0}
        >>> reassign_labels(labels, label_dict)
        array([1, 2, 0, 2, 1])
    """
    new_labels = np.zeros_like(labels)
    for key, value in label_dict.items():
        new_labels[labels == key] = value
    return new_labels


def straight_through_estimator(
    gradient: torch.Tensor, estimator: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes a straight-through estimator for a non-differentiable function.

    Args:
        gradient (torch.Tensor): The gradient tensor used for the straight-through estimator. Should have the same shape as `estimator`.
        estimator (torch.Tensor): The output tensor of a non-differentiable function.

    Returns:
        A tensor with the same shape as `estimator`, where the values are the estimator for the non-differentiable function.

    Examples::
        >>> x = torch.tensor([1., 2., 3.], requires_grad=True)
        >>> y = torch.round(x)
        >>> gradient = torch.tensor([0.5, 0.5, 0.5])
        >>> straight_through_estimator(gradient, y)
        tensor([1., 2., 3.], grad_fn=<AddBackward0>)
    """
    return stop_gradient(estimator - gradient) + gradient


def log_interpolate(
    log_a: torch.Tensor, log_b: torch.Tensor, alpha_logit: torch.Tensor
) -> torch.Tensor:
    r"""
    Log-linearly interpolates between two tensors using a logistic function.

    Args:
        log_a (torch.Tensor): The first input tensor.
        log_b (torch.Tensor): The second input tensor.
        alpha_logit (torch.Tensor): The logit of the interpolation factor alpha, defined in the range (-inf, inf).

    Returns:
        A tensor with the same shape as `log_a` and `log_b`, where each element is a log-linear interpolation of the corresponding elements in `log_a` and `log_b`.

    Examples::
        >>> log_a = torch.tensor([1., 2., 3.])
        >>> log_b = torch.tensor([4., 5., 6.])
        >>> alpha_logit = torch.tensor([0.])
        >>> log_interpolate(log_a, log_b, alpha_logit)
        tensor([1.3863, 2.3863, 3.3863])
    """
    log_alpha = F.logsigmoid(alpha_logit)
    log1m_alpha = F.logsigmoid(alpha_logit) - alpha_logit
    return torch.logaddexp(log_a + log_alpha, log_b + log1m_alpha)


def logmeanexp(
    x: torch.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim: bool = False
) -> torch.Tensor:
    r"""
    Computes the log of the mean of exponentials of input tensor along a given dimension.

    Args:
        x (torch.Tensor): The input tensor of shape :math:`(*, D)`.
        dim (int or tuple of ints, optional): The dimension or dimensions along which to perform the logmeanexp operation.
            If dim is None, logmeanexp will be calculated over all dimensions of the input tensor. Default: None.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False.

    Returns:
        The tensor with the log of the mean of exponentials of input tensor along a given dimension. If dim is None, it will be a scalar tensor.

    Examples::
        >>> x = torch.randn(2, 3, 4)
        >>> logmeanexp(x, dim=1)
        tensor([[-0.5457, -0.0862, -0.5162, -0.4415],
                [-0.2969, -0.4067, -0.2293, -0.3733]])
    """
    if dim is not None:
        if isinstance(dim, tuple):
            return x.logsumexp(dim=dim, keepdim=keepdim) - np.log(
                np.prod([x.shape[dim] for dim in dim])
            )
        else:
            return x.logsumexp(dim=dim, keepdim=keepdim) - np.log(x.shape[dim])
    else:
        return x.logsumexp(
            dim=tuple([dim for dim in range(x.dim())]), keepdim=keepdim
        ) - np.log(x.numel())


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    r"""
    Returns a flattened view of the off-diagonal elements of a square matrix.

    Args:
        x (torch.Tensor): The input tensor of shape :math:`(N, N)`.

    Returns:
        A tensor with shape :math:`((N - 1) \times N // 2,)` containing the off-diagonal elements of the input tensor.

    Examples::
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> off_diagonal(x)
        tensor([2, 3, 6])
    """
    n, m = x.shape
    assert n == m, "input tensor must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def remove_diagonal(x: torch.Tensor) -> torch.Tensor:
    r"""
    Removes the diagonal elements of a square matrix and returns the off-diagonal elements as a matrix.

    Args:
        x (torch.Tensor): The input tensor of shape :math:`(N, N)`.

    Returns:
        A tensor with shape :math:`(N, N - 1)` containing the off-diagonal elements of the input tensor.

    Examples::
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> remove_diagonal(x)
        tensor([[2, 3],
                [4, 6],
                [8, 9]])
    """
    n, m = x.shape
    return off_diagonal(x).reshape(n, n - 1)


def split_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Splits logits into positive and negative logits based on the labels.

    Args:
        logits (torch.Tensor): A 2D tensor of shape :math:`(batch\_size, num\_classes)` containing the logits for each class.
        labels (torch.Tensor): A 2D tensor of shape :math:`(batch\_size, 1)` containing the class labels for each example.

    Returns:
        A tuple containing:
        - pos_logits (torch.Tensor): A 2D tensor of shape :math:`(batch\_size, 1)` containing the logits for positive class labels.
        - neg_logits (torch.Tensor): A 2D tensor of shape :math:`(batch\_size, num\_classes-1)` containing the logits for negative class labels.

    Examples::
        >>> logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> labels = torch.tensor([[2], [1]])
        >>> split_logits(logits, labels)
        (tensor([[3.],
                 [6.]]),
         tensor([[1., 2.],
                 [4., 6.]]))
    """
    batch_size, num_classes = logits.shape

    # extract positive logits
    pos_logits = logits.gather(1, labels)

    # extract negative logits
    mask = torch.full_like(logits, True, dtype=torch.bool).scatter_(1, labels, False)
    neg_logits = logits[mask].reshape(batch_size, num_classes - 1)

    return pos_logits, neg_logits


def concat_dicts(dicts: List[Dict]) -> Dict:
    r"""
    Concatenates a list of dictionaries along the first axis.

    Args:
        dicts (List[Dict]): A list of dictionaries with the same keys and shapes.

    Returns:
        A dictionary with the same keys as the input dictionaries, where each value is a concatenated numpy array of the values from the input dictionaries.

    Examples::
        >>> dict_list = [{'a': np.array([1, 2]), 'b': np.array([3, 4])}, {'a': np.array([5, 6]), 'b': np.array([7, 8])}]
        >>> concat_dicts(dict_list)
        {'a': array([1, 2, 5, 6]), 'b': array([3, 4, 7, 8])}
    """
    return {
        key: np.concatenate([dictionary[key] for dictionary in dicts], axis=0)
        for key in dicts[0].keys()
    }


def config_to_filename(config: Dict) -> str:
    r"""
    Converts a dictionary of configuration parameters to a string for use in a filename.

    Args:
        config (Dict): A dictionary of configuration parameters.

    Returns:
        A string where each key-value pair in the input dictionary is formatted as "key=value", with each pair separated by a vertical bar (|).

    Examples::
        >>> config = {'batch_size': 64, 'learning_rate': 0.001, 'dropout': 0.5}
        >>> config_to_filename(config)
        'batch_size=64|learning_rate=0.001|dropout=0.5'
    """
    items = [f"{k}={v}" for k, v in config.items()]
    return "|".join(items)


class Trainer(pl.Trainer):
    def predict(
        self,
        model=None,
        dataloaders=None,
        datamodule=None,
        return_predictions=None,
        ckpt_path=None,
    ) -> Dict:
        r"""
        Runs inference on a trained model and returns the predictions as a concatenated dictionary.

        Args:
            model: Model to run inference on. If None, uses the trainer's current model.
            dataloaders: List of PyTorch dataloaders or None. If None, uses the dataloader(s) associated with the trainer's current datamodule.
            datamodule: PyTorch Lightning datamodule or None. If None, uses the trainer's current datamodule.
            return_predictions: bool or None. If True, returns the predictions. If None, defaults to the trainer's current setting.
            ckpt_path: str or None. Path to a checkpoint file to use for inference.

        Returns:
            A dictionary containing the concatenated predictions from running inference on the model.

        Examples::
            >>> trainer = Trainer()
            >>> predictions = trainer.predict(model=model, datamodule=datamodule)
            >>> print(predictions)
            {'output_1': array([0.1, 0.2, 0.3]), 'output_2': array([0.4, 0.5, 0.6])}
        """
        result = super().predict(
            model=model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            return_predictions=return_predictions,
            ckpt_path=ckpt_path,
        )
        return concat_dicts(result)
