# Copyright 2025 Hackable Diffusion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization layers.

Implements the following methods:
- RMSNorm: https://arxiv.org/abs/1910.07467
- GroupNorm: https://arxiv.org/abs/1803.08494
"""

import einops
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp


################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

NormalizationType = arch_typing.NormalizationType

################################################################################
# MARK: NormalizationLayerFactory
################################################################################


class NormalizationLayerFactory:
  """A factory for creating normalization layers.

  This class provides a convenient way to configure and create
  `NormalizationLayer` instances. It separates the configuration of the
  normalization from its application, allowing for easy injection of different
  normalization strategies.

  It can create both conditional and unconditional normalization layers via
  the `conditional_norm_factory` and `unconditional_norm_factory` properties.

  Attributes:
    normalization_method: The normalization method to use (e.g., 'rms_norm').
    num_groups: The number of groups to use for group normalization. If None,
      group normalization cannot be used and an error will be raised.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: The data type of the computation.
  """

  def __init__(
      self,
      normalization_method: NormalizationType,
      num_groups: int | None = None,
      epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
  ):
    self.normalization_method = normalization_method
    self.epsilon = epsilon
    self.num_groups = num_groups
    self.dtype = dtype

  @property
  def unconditional_norm_factory(self):
    """Returns a factory for creating unconditional normalization layers."""
    return lambda: NormalizationLayer(
        normalization_method=self.normalization_method,
        conditional=False,
        num_groups=self.num_groups,
        epsilon=self.epsilon,
        name="UnconditionalNorm",
        dtype=self.dtype,
    )

  @property
  def conditional_norm_factory(self):
    """Returns a factory for creating conditional normalization layers."""
    return lambda: NormalizationLayer(
        normalization_method=self.normalization_method,
        conditional=True,
        num_groups=self.num_groups,
        epsilon=self.epsilon,
        name="ConditionalNorm",
        dtype=self.dtype,
    )


################################################################################
# MARK: NormalizationLayer
################################################################################


class NormalizationLayer(nn.Module):
  """A generic normalization layer with optional conditioning.

  This layer applies a specified normalization method to the input tensor `x`.
  If `conditional` is True, it then applies a learned scale and shift
  transformation conditioned on an embedding `c`.

  The scale and shift are computed from conditioning `c` using a dense layer.

  Attributes:
    normalization_method: The normalization method to use.
    conditional: Whether to apply conditional scaling and shifting.
    num_groups: The number of groups to use for group normalization. If None,
      group normalization cannot be used and an error will be raised.
    epsilon: Epsilon value for numerical stability in normalization.
    dtype: The data type of the computation.
  """

  normalization_method: NormalizationType
  conditional: bool
  num_groups: int | None = None
  epsilon: float = 1e-5
  dtype: DType = jnp.float32

  def setup(self):
    if (
        self.normalization_method == NormalizationType.GROUP_NORM
        and self.num_groups is None
    ):
      raise ValueError("num_groups must be specified for Group normalization.")

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch ... channels"],
      c: Float["batch cond_dim"] | None = None,
  ) -> Float["batch ... channels"]:
    x_shape = x.shape
    ch = x_shape[-1]  # (B ... channel)

    if self.normalization_method == NormalizationType.RMS_NORM:
      x = nn.RMSNorm(
          epsilon=self.epsilon,
          dtype=self.dtype,
          reduction_axes=-1,  # for (B ... ch) results in (B ... ) rms values
          feature_axes=-1,  # per channel learnable scale
      )(x)
    elif self.normalization_method == NormalizationType.GROUP_NORM:
      x = nn.GroupNorm(
          epsilon=self.epsilon,
          dtype=self.dtype,
          reduction_axes=None,  # reduction over (H, W, C)
          num_groups=self.num_groups,
      )(x)
    else:
      raise ValueError(
          f"Unsupported normalization method: {self.normalization_method}"
      )

    if self.conditional:

      scale_and_shift = nn.Dense(
          ch * 2,
          kernel_init=nn.zeros_init(),
          bias_init=nn.zeros_init(),
          dtype=self.dtype,
      )(c)
      scale, shift = jnp.split(scale_and_shift, 2, axis=-1)  # (B, channel) each

      x = einops.rearrange(x, "b ... c -> b c ...")  # (B, channel, ...)
      scale = utils.bcast_right(scale, x.ndim)
      shift = utils.bcast_right(shift, x.ndim)
      x = (1.0 + scale) * x + shift
      x = einops.rearrange(x, "b c ... -> b ... c")

    return x
