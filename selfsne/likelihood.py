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
import torch.distributions as D
from torch.nn import Module


def split(x, num_splits=2, dim=-1):
    return torch.split(x, x.shape[dim] // num_splits, dim=dim)


class KLDLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(KLDLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        p = F.softmax(x.log(), dim=self.reduction_dim)
        log_p = F.log_softmax(x.log(), dim=self.reduction_dim)
        log_q = F.log_softmax(x_hat, dim=self.reduction_dim)
        cross_entropy = -(p * log_q).sum(self.reduction_dim)
        entropy = (p * log_p).sum(self.reduction_dim)
        return entropy + cross_entropy


class BDLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(BDLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        log_p = F.log_softmax(x.log(), dim=self.reduction_dim)
        log_q = F.log_softmax(x_hat, dim=self.reduction_dim)
        log_bc = -0.5 * (log_p + log_q).logsumexp(self.reduction_dim)
        return log_bc


class CrossEntropyLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(CrossEntropyLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        p = F.softmax(x.log(), dim=self.reduction_dim)
        log_q = F.log_softmax(x_hat, dim=self.reduction_dim)
        return -(p * log_q).sum(self.reduction_dim)


class CrossEntropyMILoss(Module):
    def __init__(self, reduction_dim=-1):
        super(CrossEntropyMILoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        n = x.shape[0]
        p = F.softmax(x.log(), dim=self.reduction_dim)
        log_q = F.log_softmax(x_hat, dim=self.reduction_dim)
        cross_entropy = torch.matmul(p, log_q.T)
        eye = torch.eye(n, device=x.device)
        similarity = -(eye * cross_entropy).sum(-1)
        contrastive = (cross_entropy).logsumexp(-1) - np.log(n)
        return similarity + contrastive


class BinaryCrossEntropyLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, p, x_hat):
        log_q = F.logsigmoid(x_hat)
        crossentropy = -(p * log_q).mean(self.reduction_dim)
        log_q = F.logsigmoid(-x_hat)
        crossentropy_inv = -((1 - p) * log_q).mean(self.reduction_dim)
        return crossentropy + crossentropy_inv


class MSELoss(Module):
    def __init__(self, reduction_dim=-1):
        super(MSELoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        return (x - x_hat).pow(2).sum(self.reduction_dim)


class MAELoss(Module):
    def __init__(self, reduction_dim=-1):
        super(MSELoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        return (x - x_hat).abs().sum(self.reduction_dim)


class NormalLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(NormalLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 2

    def forward(self, x, x_hat):
        loc, logscale = split(x_hat, self.multiplier, self.reduction_dim)
        scale = F.softplus(logscale) + 1e-8
        log_likelihood = D.Normal(loc, scale).log_prob(x)
        return -log_likelihood.sum(self.reduction_dim)


class LaplaceLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(LaplaceLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 2

    def forward(self, x, x_hat):
        loc, logscale = split(x_hat, self.multiplier, self.reduction_dim)
        scale = F.softplus(logscale) + 1e-8
        log_likelihood = D.Laplace(loc, scale).log_prob(x)
        return -log_likelihood.sum(self.reduction_dim)


class StudentTLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(StudentTLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 3

    def forward(self, x, x_hat):
        loc, logscale, logdf = split(x_hat, self.multiplier, self.reduction_dim)
        scale = F.softplus(logscale) + 1e-8
        df = F.softplus(logdf) + 1e-8
        log_likelihood = D.StudentT(df, loc, scale).log_prob(x)
        return -log_likelihood.sum(self.reduction_dim)


class BetaLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(BetaLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 2

    def forward(self, x, x_hat):
        logmu, logphi = split(x_hat, self.multiplier, self.reduction_dim)
        log_A = F.logsigmoid(logmu) + logphi
        log_B = F.logsigmoid(-logmu) + logphi
        log_likelihood = D.Beta(log_A.exp(), log_B.exp()).log_prob(x)
        return -log_likelihood.sum(self.reduction_dim)


class DirichletLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(DirichletLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        return -D.Dirichlet(F.softplus(x_hat)).log_prob(x)


class GammaLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(GammaLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 2

    def forward(self, x, x_hat):
        logmu, logphi = split(x_hat, self.multiplier, self.reduction_dim)
        log_A = (logmu + logmu) - logphi
        log_B = logmu - logphi
        log_likelihood = D.Gamma(log_A.exp(), log_B.exp()).log_prob(x)
        return -log_likelihood.sum(self.reduction_dim)


class CosineLoss(Module):
    def __init__(self, reduction_dim=-1):
        super(CosineLoss, self).__init__()
        self.reduction_dim = -1
        self.multiplier = 1

    def forward(self, x, x_hat):
        x_hat = x_hat / x_hat.norm(dimß=self.reduction_dim, keepdim=True)
        x = x / x.norm(dim=self.reduction_dim, keepdim=True)
        return -(x * x_hat).sum(self.reduction_dim)


LIKELIHOODS = {
    "kld": KLDLoss,
    "bd": BDLoss,
    "cross_entropy": CrossEntropyLoss,
    "mi_cross_entropy": CrossEntropyMILoss,
    "softmax": CrossEntropyLoss,
    "categorical": CrossEntropyLoss,
    "binary": BinaryCrossEntropyLoss,
    "bernoulli": BinaryCrossEntropyLoss,
    "beta": BetaLoss,
    "gamma": GammaLoss,
    "mse": MSELoss,
    "mae": MAELoss,
    "normal": NormalLoss,
    "studentt": StudentTLoss,
    "laplace": LaplaceLoss,
    "cosine": CosineLoss,
    "dirichlet": DirichletLoss,
}
