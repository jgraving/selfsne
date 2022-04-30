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


class EMA:
    """Exponential Moving Average"""

    def __init__(self, beta=0.9):
        self.beta = beta

    def __call__(self, moving_average, value):
        return moving_average * self.beta + (1 - self.beta) * value


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
