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
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList, Dropout, AlphaDropout
from torch.nn import init
import numpy as np


def lecun_normal_(x):
    return init.normal_(x, std=np.sqrt(1 / x.shape[1]))


def init_selu(x: Linear):
    lecun_normal_(x.weight)
    init.zeros_(x.bias)


def init_linear(x: Linear):
    init.xavier_uniform_(x.weight)
    init.zeros_(x.bias)


activations = {"selu": F.selu, "elu": F.elu, "relu": F.relu}


class FeedForward(Module):
    def __init__(
        self, in_features, layers=[256, 256, 256], activation="selu", dropout=0.0
    ):
        super(FeedForward, self).__init__()
        if activation in activations.keys():
            self.activation = activations[activation]
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                "this activation function is not supported "
                "must be 'selu', 'relu', 'elu', or callable"
            )

        linears = [Linear(in_features, layers[0])]
        for idx in range(1, len(layers)):
            linears.append(Linear(layers[idx - 1], layers[idx]))
        init = init_selu if activation in ["selu", F.selu] else init_linear
        for linear in linears:
            init(linear)
        self.linears = ModuleList(linears)
        self.dropout = Dropout(
            dropout
        )  # if self.activation != "selu" else AlphaDropout(dropout)

    def forward(self, x):
        for linear in self.linears:
            x = self.activation(linear(x))
            x = self.dropout(x)
        return x
