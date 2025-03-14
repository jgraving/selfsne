# nn/init.py
# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import init

__all__ = [
    "lecun_normal_",
    "init_linear",
]


def lecun_normal_(x, mode="fan_in"):
    """
    Initialize tensor x using LeCun's normal initialization
    via Kaiming normal initialization with linear nonlinearity.
    """
    return init.kaiming_normal_(x, mode=mode, nonlinearity="linear")


def init_linear(x):
    """
    Initialize a linear layer by applying LeCun's normal initialization
    to its weights and zeroing out its biases.
    """
    lecun_normal_(x.weight)
    if hasattr(x, "bias") and x.bias is not None:
        init.zeros_(x.bias)
    return x
