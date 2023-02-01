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


def stop_gradient(x):
    return x.clone().detach()


def set_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def random_sample_columns(x, num_samples):
    idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).repeat(1, num_samples)
    jdx = torch.randint(0, x.shape[1], (x.shape[0], num_samples), device=x.device)
    return x[idx, jdx]


def disable_grad(module):
    module.eval()
    set_grad(module, False)


def enable_grad(module):
    module.train()
    set_grad(module, True)


def reassign_labels(labels, label_dict):
    # Create a new array of the same shape as labels
    new_labels = np.zeros_like(labels)

    for key, value in label_dict.items():
        # Assign new labels based on the dictionary
        new_labels[labels == key] = value
    return new_labels


def straight_through_estimator(gradient, estimator):
    return stop_gradient(estimator - gradient) + gradient


def log_interpolate(log_a, log_b, alpha_logit):
    log_alpha = F.logsigmoid(alpha_logit)
    log1m_alpha = F.logsigmoid(alpha_logit) - alpha_logit
    return torch.logaddexp(
        log_a + log_alpha,
        log_b + log1m_alpha,
    )


def logmeanexp(x, dim=None, keepdim=False):
    if dim is not None:
        if isinstance(dim, tuple):
            return x.logsumexp(dim=dim, keepdim=keepdim) - np.log(
                np.sum([x.shape[dim] for dim in dim])
            )
        else:
            return x.logsumexp(dim=dim, keepdim=keepdim) - np.log(np.sum(x.shape[dim]))
    else:
        return x.logsumexp(
            dim=tuple([dim for dim in range(x.dim())]), keepdim=keepdim
        ) - np.log(x.numel())


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def remove_diagonal(x):
    # remove diagonal elements of a n x n matrix
    # and return off diagonal elements as n x (n - 1) matrix
    n, m = x.shape
    return off_diagonal(x).reshape(n, n - 1)


def concat_dicts(dicts):
    return {
        key: np.concatenate([dictionary[key] for dictionary in dicts], axis=0)
        for key in dicts[0].keys()
    }


class Trainer(pl.Trainer):
    def predict(
        self,
        model=None,
        dataloaders=None,
        datamodule=None,
        return_predictions=None,
        ckpt_path=None,
    ):
        result = super().predict(
            model=model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            return_predictions=return_predictions,
            ckpt_path=ckpt_path,
        )
        return concat_dicts(result)
