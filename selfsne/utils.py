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


def log_interpolate(log_a, log_b, alpha_logit):
    log_alpha = F.logsigmoid(alpha_logit)
    log1m_alpha = F.logsigmoid(alpha_logit) - alpha_logit
    return torch.logaddexp(
        log_a + log_alpha,
        log_b + log1m_alpha,
    )


def logmeanexp(x, dim=None):
    if dim is not None:
        return x.logsumexp(dim) - np.log(x.shape[dim])
    else:
        return x.logsumexp(tuple([dim for dim in range(x.dim())])) - np.log(x.numel())


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
