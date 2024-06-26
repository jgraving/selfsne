# -*- coding: utf-8 -*-
# Copyright 2020 Jacob M. Graving <jgraving@gmail.com>
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
from torch import nn
from torch.nn import init

import numpy as np

from selfsne.nn import init_selu
from selfsne.kernels import pairwise_inner_product
from selfsne.utils import (
    off_diagonal,
    remove_diagonal,
    logmeanexp,
    log_interpolate,
    stop_gradient,
)

from typing import Optional, List, Union


class LogMeanExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, log_moving_average, momentum_logit):
        log_batch_average = logmeanexp(input)
        updated_log_moving_average = stop_gradient(
            log_interpolate(log_moving_average, log_batch_average, momentum_logit)
        )
        log_moving_average.data = updated_log_moving_average
        ctx.save_for_backward(input, updated_log_moving_average)
        return updated_log_moving_average

    @staticmethod
    def backward(ctx, grad_output):
        input, log_moving_average = ctx.saved_tensors
        log_n = np.log(input.numel())
        # ratio = input.exp() / log_moving_average.add(log_n).exp()
        # Compute the ratio in the log domain
        ratio = torch.exp(input - (log_moving_average + log_n))
        return grad_output * ratio, None, None


class MomentumBaseline(nn.Module):
    def __init__(
        self,
        momentum: float = 0.9,
    ) -> None:
        """
        Calculates the momentum baseline for a given set of logits.

        Args:
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
        """
        super().__init__()
        self.register_buffer("log_moving_average", torch.zeros(1))
        self.register_buffer("momentum_logit", torch.zeros(1).add(momentum).logit())

    def forward(self, neg_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the momentum baseline for a given set of logits.

        Args:
            neg_logits (torch.Tensor): Negative logits tensor.

        Returns:
            torch.Tensor: Momentum baseline for the given set of logits.
        """
        if self.training:
            return LogMeanExp.apply(
                neg_logits,
                self.log_moving_average,
                self.momentum_logit,
            )
        else:
            return self.log_moving_average


class ReverseMomentumBaseline(MomentumBaseline):
    def forward(self, pos_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the reverse momentum baseline for a given set of logits.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.

        Returns:
            torch.Tensor: Momentum baseline for the given set of logits.
        """
        if self.training:
            return -LogMeanExp.apply(
                -pos_logits,
                self.log_moving_average,
                self.momentum_logit,
            )
        else:
            return -self.log_moving_average


class BatchBaseline(nn.Module):
    def forward(
        self,
        neg_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the batch baseline for a given set of logits.

        Args:
            neg_logits (torch.Tensor): Negative logits tensor.

        Returns:
            torch.Tensor: Batch baseline for the given set of logits.
        """
        return logmeanexp(neg_logits)


class ReverseBatchBaseline(nn.Module):
    def forward(
        self,
        pos_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the reverse batch baseline for a given set of logits.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.

        Returns:
            torch.Tensor: Batch baseline for the given set of logits.
        """
        return -logmeanexp(-pos_logits)


class BatchConditionalBaseline(nn.Module):
    def forward(
        self,
        neg_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the batch conditional baseline for a given set of logits.

        Args:
            neg_logits (torch.Tensor): Negative logits tensor.

        Returns:
            torch.Tensor: Batch conditional baseline for the given set of logits.
        """
        return logmeanexp(neg_logits, dim=-1, keepdim=True)


class ReverseBatchConditionalBaseline(nn.Module):
    def forward(
        self,
        pos_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the reverse batch conditional baseline for a given set of logits.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.

        Returns:
            torch.Tensor: Batch conditional baseline for the given set of logits.
        """
        return -logmeanexp(-pos_logits, dim=-1, keepdim=True)


class ParametricBaseline(nn.Module):
    def __init__(
        self,
        param_dim: int = 1024,
        activation: Optional[nn.Module] = nn.Identity(),
        bias_init: float = 0.0,  # Default bias initialization value
    ):
        """
        Initializes a parametric baseline module.

        Args:
            activation (nn.Module, optional): Activation function to apply to the output. If None is passed, uses nn.Identity. Defaults to nn.Identity().
        """
        super().__init__()
        self.param = nn.Parameter(torch.randn(1, param_dim))
        self.projection = init_selu(nn.Linear(param_dim, 1))
        init.constant_(
            self.projection.bias, bias_init
        )  # Initializing bias to bias_init
        self.activation = activation if activation is not None else nn.Identity()

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns the parametric baseline.

        Returns:
            torch.Tensor: Parametric baseline.
        """
        return self.activation(self.projection(self.param))


class ConstantBaseline(nn.Module):
    def __init__(self, baseline: float = 0.0):
        """
        Initializes a constant baseline module.

        Args:
            baseline (float, optional): Constant baseline value. Defaults to 0.0.
        """
        super().__init__()
        self.register_buffer("baseline", torch.tensor(baseline))

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the constant baseline for a given set of logits.


        Returns:
            torch.Tensor: Constant baseline for the given set of logits.
        """
        return self.baseline


class ParametricConditionalBaseline(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        param_dim: int = 1024,
        activation: Optional[nn.Module] = nn.Identity(),
        bias_init: float = 0.0,  # Default bias initialization value
    ):
        """
        Initializes a parametric conditional baseline module.

        Args:
            encoder (nn.Module): Encoder module to encode the input tensor z_x.
            activation (nn.Module, optional): Activation function to apply to the output. Defaults to nn.Identity().
        """
        super().__init__()
        self.encoder = encoder
        self.parametric_baseline = ParametricBaseline(param_dim, bias_init=bias_init)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(
        self,
        z_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the parametric conditional baseline for a given set of data x or embedded data z_x.

        Args:
            x (torch.Tensor, optional): Data input to the encoder. Ignored if embedding_input=True.
            z_x (torch.Tensor, optional): Embedded data input to the encoder. Used if embedding_input=True.

        Returns:
            torch.Tensor: Parametric conditional baseline for the data x or z_x.
        """
        encoded = self.encoder(z_x)

        scalar_baseline = self.parametric_baseline()

        return self.activation(scalar_baseline + encoded)


ConditionalBaseline = ParametricConditionalBaseline


class BilinearBaseline(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        param_dim: int = 1024,
        activation: Optional[nn.Module] = nn.Identity(),
        bias_init: float = 0.0,  # Default bias initialization value
    ):
        """
        Initializes a bilinear baseline module.

        Args:
            encoder (nn.Module): Encoder module to encode the input tensors z_x and z_y.
            activation (nn.Module, optional): Activation function to apply to the output. Defaults to nn.Identity().
        """
        super().__init__()
        self.encoder = encoder
        self.parametric_baseline = ParametricBaseline(param_dim, bias_init=bias_init)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(
        self,
        z_x: Optional[torch.Tensor] = None,
        z_y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the bilinear baseline for given sets of embedded data z_x and z_y.

        Args:
            z_x (torch.Tensor, optional): Embedded data input to the encoder.
            z_y (torch.Tensor, optional): Another set of embedded data input to the encoder.

        Returns:
            torch.Tensor: Bilinear baseline for the data z_x and z_y.
        """
        if z_x is None or z_y is None:
            raise ValueError("Both z_x and z_y must be provided")

        encoded_x = self.encoder(z_x)
        encoded_y = self.encoder(z_y)

        dot_product = pairwise_inner_product(encoded_x, encoded_y)

        scalar_baseline = self.parametric_baseline()

        output = dot_product + scalar_baseline

        return self.activation(output)


class MixtureBaseline(nn.Module):
    def __init__(
        self, baseline_a: nn.Module, baseline_b: nn.Module, alpha: float = 0.5
    ):
        """
        Initializes a mixture baseline module.

        Args:
            baseline_a (nn.Module): First baseline module.
            baseline_b (nn.Module): Second baseline module.
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__()
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b
        self.alpha = alpha
        self.register_buffer("alpha_logit", torch.tensor(alpha).logit())


class ArithmeticMixtureBaseline(MixtureBaseline):
    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        z_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the mixture baseline.

        Args:
            pos_logits (torch.Tensor, optional): Positive logits tensor.
            neg_logits (torch.Tensor, optional): Negative logits tensor.
            x (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_x (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Mixture baseline for the given set of logits and optional data x or latents z_x.
        """
        return log_interpolate(
            self.baseline_a(pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x),
            self.baseline_b(pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x),
            self.alpha_logit,
        )


LogInterpolatedBaseline = ArithmeticMixtureBaseline


class HarmonicMixtureBaseline(MixtureBaseline):
    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        z_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the mixture baseline.

        Args:
            pos_logits (torch.Tensor, optional): Positive logits tensor.
            neg_logits (torch.Tensor, optional): Negative logits tensor.
            x (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_x (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Mixture baseline for the given set of logits and optional data x or latents z_x.
        """
        return -log_interpolate(
            -self.baseline_a(
                pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x
            ),
            -self.baseline_b(
                pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x
            ),
            self.alpha_logit,
        )


class GeometricMixtureBaseline(MixtureBaseline):
    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        z_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the mixture baseline.

        Args:
            pos_logits (torch.Tensor, optional): Positive logits tensor.
            neg_logits (torch.Tensor, optional): Negative logits tensor.
            x (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_x (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Mixture baseline for the given set of logits and optional data x or latents z_x.
        """
        return self.alpha * self.baseline_a(
            pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x
        ) + (1 - self.alpha) * self.baseline_b(
            pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x
        )


InterpolatedBaseline = GeometricMixtureBaseline


class GeometricMomentumBaseline(GeometricMixtureBaseline):
    def __init__(self, alpha: float = 0.5, momentum: float = 0.9) -> None:
        """
        Initializes a geometric mixture of momentum baselines with specified alpha and momentum.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
        """
        super().__init__(
            baseline_a=ReverseMomentumBaseline(momentum=momentum),
            baseline_b=MomentumBaseline(momentum=momentum),
            alpha=alpha,
        )


class HarmonicMomentumBaseline(HarmonicMixtureBaseline):
    def __init__(self, alpha: float = 0.5, momentum: float = 0.9) -> None:
        """
        Initializes a harmonic mixture of momentum baselines with specified alpha and momentum.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
        """
        super().__init__(
            baseline_a=ReverseMomentumBaseline(momentum=momentum),
            baseline_b=MomentumBaseline(momentum=momentum),
            alpha=alpha,
        )


class ArithmeticMomentumBaseline(ArithmeticMixtureBaseline):
    def __init__(self, alpha: float = 0.5, momentum: float = 0.9) -> None:
        """
        Initializes an arithmetic mixture of momentum baselines with specified alpha and momentum.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
            momentum (float, optional): Momentum value for moving average calculation. Defaults to 0.9.
        """
        super().__init__(
            baseline_a=ReverseMomentumBaseline(momentum=momentum),
            baseline_b=MomentumBaseline(momentum=momentum),
            alpha=alpha,
        )


class GeometricBatchBaseline(GeometricMixtureBaseline):
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes a geometric mixture of batch baselines with specified alpha.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__(
            baseline_a=ReverseBatchBaseline(), baseline_b=BatchBaseline(), alpha=alpha
        )


class HarmonicBatchBaseline(HarmonicMixtureBaseline):
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes a harmonic mixture of batch baselines with specified alpha.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__(
            baseline_a=ReverseBatchBaseline(), baseline_b=BatchBaseline(), alpha=alpha
        )


class ArithmeticBatchBaseline(ArithmeticMixtureBaseline):
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes an arithmetic mixture of batch baselines with specified alpha.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__(
            baseline_a=ReverseBatchBaseline(), baseline_b=BatchBaseline(), alpha=alpha
        )


class GeometricBatchConditionalBaseline(GeometricMixtureBaseline):
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes a geometric mixture of batch conditional baselines with specified alpha.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__(
            baseline_a=ReverseBatchConditionalBaseline(),
            baseline_b=BatchConditionalBaseline(),
            alpha=alpha,
        )


class HarmonicBatchConditionalBaseline(HarmonicMixtureBaseline):
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes a harmonic mixture of batch conditional baselines with specified alpha.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__(
            baseline_a=ReverseBatchConditionalBaseline(),
            baseline_b=BatchConditionalBaseline(),
            alpha=alpha,
        )


class ArithmeticBatchConditionalBaseline(ArithmeticMixtureBaseline):
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes an arithmetic mixture of batch conditional baselines with specified alpha.

        Args:
            alpha (float, optional): Weighting parameter for interpolation. Defaults to 0.5.
        """
        super().__init__(
            baseline_a=ReverseBatchConditionalBaseline(),
            baseline_b=BatchConditionalBaseline(),
            alpha=alpha,
        )


class AdditiveBaseline(nn.Module):
    def __init__(self, baselines: List[nn.Module], mean: bool = False):
        """
        Initializes an additive baseline module.

        Args:
            baselines (List[nn.Module]): List of baseline modules to be added together.
            mean (bool, optional): Whether to take the mean of the baselines instead of summing them. Defaults to False.
        """
        super().__init__()
        self.baselines = nn.ModuleList(baselines)
        self.scale = len(baselines) if mean else 1
        self.log_scale = np.log(self.scale)

    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        z_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the additive baseline.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.
            neg_logits (torch.Tensor): Negative logits tensor.
            x (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_x (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Additive baseline for the given set of logits and optional data x.
        """
        log_baseline = 0
        for baseline in self.baselines:
            log_baseline = log_baseline + baseline(
                pos_logits=pos_logits, neg_logits=neg_logits, x=x, z_x=z_x
            )
        return log_baseline / self.scale


class LogAdditiveBaseline(AdditiveBaseline):
    def __init__(self, baselines: List[nn.Module], mean: bool = False):
        """
        Initializes a logarithmic additive baseline module.

        Args:
            baselines (List[nn.Module]): List of baseline modules to be added together.
            mean (bool, optional): Whether to take the mean of the baselines instead of summing them. Defaults to False.
        """
        super().__init__(baselines, mean)

    def forward(
        self,
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        z_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the logarithmic additive baseline for a given set of logits and optional data x.

        Args:
            pos_logits (torch.Tensor): Positive logits tensor.
            neg_logits (torch.Tensor): Negative logits tensor.
            x (torch.Tensor, optional): Data for LearnedConditionalBaseline. Defaults to None.
            z_x (torch.Tensor, optional): Embedded data for LearnedConditionalBaseline. Defaults to None.

        Returns:
            torch.Tensor: Logarithmic additive baseline for the given set of logits and optional data x.
        """
        # Set the initial value from the first baseline
        log_baseline = self.baselines[0](pos_logits=pos_logits, x=x, z_x=z_x)

        for baseline in self.baselines[1:]:
            additional_value = baseline(pos_logits=pos_logits, x=x, z_x=z_x)
            log_baseline = torch.logaddexp(log_baseline, additional_value)

        return log_baseline - self.log_scale


BASELINES = {
    "batch": BatchBaseline,
    "batch_conditional": BatchConditionalBaseline,
    "momentum": MomentumBaseline,
    "reverse_batch": ReverseBatchBaseline,
    "reverse_batch_conditional": ReverseBatchConditionalBaseline,
    "reverse_momentum": ReverseMomentumBaseline,
    "geometric_momentum": GeometricMomentumBaseline,
    "harmonic_momentum": HarmonicMomentumBaseline,
    "arithmetic_momentum": ArithmeticMomentumBaseline,
    "geometric_batch": GeometricBatchBaseline,
    "harmonic_batch": HarmonicBatchBaseline,
    "arithmetic_batch": ArithmeticBatchBaseline,
    "geometric_batch_conditional": GeometricBatchConditionalBaseline,
    "harmonic_batch_conditional": HarmonicBatchConditionalBaseline,
    "arithmetic_batch_conditional": ArithmeticBatchConditionalBaseline,
    "parametric": ParametricBaseline,
    "constant": ConstantBaseline,
}


def baseline_factory(baseline_key, divergence_key):
    if baseline_key in ["parametric", "constant"]:
        modified_baseline_key = baseline_key
    else:
        divergence_to_prefix = {
            "kld": "",  # kld baselines use no prefix
            "rkld": "reverse_",
            "jsd": "geometric_",
        }

        prefix = divergence_to_prefix.get(divergence_key)
        if prefix is None:
            raise ValueError(f"Unsupported divergence key: {divergence_key}")

        modified_baseline_key = f"{prefix}{baseline_key}"

    baseline_class = BASELINES.get(modified_baseline_key)
    if not baseline_class:
        raise ValueError(f"No baseline found for key: {modified_baseline_key}")

    return baseline_class
