# Copyright 2026 Hackable Diffusion Authors.
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

"""Normalization strategies and layers.

Implements explicit conditional and unconditional strategies for:
- RMSNorm: https://arxiv.org/abs/1910.07467
- GroupNorm: https://arxiv.org/abs/1803.08494
- LayerNorm: https://arxiv.org/abs/1607.06450

Note: scale/shift are (B, C) and x is channels-last (B, ..., C),
so we insert singleton spatial dims to broadcast over spatial axes.
"""

import dataclasses
from typing import Union

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType

################################################################################
# MARK: Private Modules
################################################################################


class _ConditioningWithTransform(nn.Module):
  """Implements conditioning with potential scale and shift modulation."""

  use_shift: bool = True
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
    ch = x.shape[-1]
    spatial_dims = x.ndim - 2
    broadcast_shape = (x.shape[0], *[1] * spatial_dims, ch)

    if self.use_shift:
      scale_and_shift = nn.Dense(
          ch * 2,
          kernel_init=nn.initializers.zeros_init(),
          bias_init=nn.initializers.zeros_init(),
          dtype=self.dtype,
          name="Dense_ScaleShift",
      )(c)
      scale, shift = jnp.split(scale_and_shift, 2, axis=-1)
      return (1.0 + scale.reshape(broadcast_shape)) * x + shift.reshape(
          broadcast_shape
      )
    else:
      scale = nn.Dense(
          ch,
          kernel_init=nn.initializers.zeros_init(),
          bias_init=nn.initializers.zeros_init(),
          dtype=self.dtype,
          name="Dense_Scale",
      )(c)
      return (1.0 + scale.reshape(broadcast_shape)) * x


class _SafeGroupNorm(nn.GroupNorm):
  """GroupNorm that validates the mask shape to prevent JAX reshape errors."""

  @nn.compact
  def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    if mask is not None and mask.shape[-1] != x.shape[-1]:
      raise ValueError(
          "If using GroupNorm with a mask, the mask's last dimension must"
          " match the input's channel dimension. Otherwise, one cannot"
          " reshape the mask during the grouping operation."
      )
    return super().__call__(x, mask=mask)


class _NormalizationLayer(nn.Module):
  """Orchestrates a base normalizer and optional conditioning."""

  base_norm: nn.Module
  conditioning: nn.Module | None = None

  @nn.compact
  @kt.typechecked
  def __call__(
      self,
      x: jax.Array,
      c: jax.Array | None = None,
      *,
      mask: jax.Array | None = None,
  ) -> jax.Array:
    x = self.base_norm(x, mask=mask)

    if self.conditioning is not None:
      if c is None:
        raise ValueError(
            "Conditioning 'c' must be provided for a conditional norm layer."
        )
      x = self.conditioning(x, c)

    return x


################################################################################
# MARK: Normalization Strategies (Configuration Mappings)
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class RMSNormStrategy:
  """Unconditional RMS Normalization strategy."""

  epsilon: float = 1e-5
  use_scale: bool = True
  dtype: DType = jnp.float32

  def build_layer(self, name: str | None = None) -> nn.Module:
    return _NormalizationLayer(
        base_norm=nn.RMSNorm(
            epsilon=self.epsilon,
            use_scale=self.use_scale,
            dtype=self.dtype,
        ),
        conditioning=None,
        name=name or "RMSNorm",
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalRMSNormStrategy:
  """Conditional RMS Normalization strategy."""

  epsilon: float = 1e-5
  use_shift: bool = True
  dtype: DType = jnp.float32

  def build_layer(self, name: str | None = None) -> nn.Module:
    return _NormalizationLayer(
        base_norm=nn.RMSNorm(
            epsilon=self.epsilon,
            use_scale=False,
            dtype=self.dtype,
        ),
        conditioning=_ConditioningWithTransform(
            use_shift=self.use_shift, dtype=self.dtype
        ),
        name=name or "ConditionalRMSNorm",
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class LayerNormStrategy:
  """Unconditional Layer Normalization strategy."""

  epsilon: float = 1e-5
  use_bias: bool = True
  use_scale: bool = True
  dtype: DType = jnp.float32

  def build_layer(self, name: str | None = None) -> nn.Module:
    return _NormalizationLayer(
        base_norm=nn.LayerNorm(
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            reduction_axes=-1,
            feature_axes=-1,
            dtype=self.dtype,
        ),
        conditioning=None,
        name=name or "LayerNorm",
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalLayerNormStrategy:
  """Conditional Layer Normalization strategy."""

  epsilon: float = 1e-5
  use_shift: bool = True
  dtype: DType = jnp.float32

  def build_layer(self, name: str | None = None) -> nn.Module:
    return _NormalizationLayer(
        base_norm=nn.LayerNorm(
            epsilon=self.epsilon,
            use_bias=False,
            use_scale=False,
            reduction_axes=-1,
            feature_axes=-1,
            dtype=self.dtype,
        ),
        conditioning=_ConditioningWithTransform(
            use_shift=self.use_shift, dtype=self.dtype
        ),
        name=name or "ConditionalLayerNorm",
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GroupNormStrategy:
  """Unconditional Group Normalization strategy."""

  num_groups: int
  epsilon: float = 1e-5
  use_bias: bool = True
  use_scale: bool = True
  dtype: DType = jnp.float32

  def build_layer(self, name: str | None = None) -> nn.Module:
    return _NormalizationLayer(
        base_norm=_SafeGroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            reduction_axes=None,
            dtype=self.dtype,
        ),
        conditioning=None,
        name=name or "GroupNorm",
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalGroupNormStrategy:
  """Conditional Group Normalization strategy."""

  num_groups: int
  epsilon: float = 1e-5
  use_shift: bool = True
  dtype: DType = jnp.float32

  def build_layer(self, name: str | None = None) -> nn.Module:
    return _NormalizationLayer(
        base_norm=_SafeGroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=False,
            use_scale=False,
            reduction_axes=None,
            dtype=self.dtype,
        ),
        conditioning=_ConditioningWithTransform(
            use_shift=self.use_shift, dtype=self.dtype
        ),
        name=name or "ConditionalGroupNorm",
    )


################################################################################
# MARK: Strategy Type Unions
################################################################################

UnconditionalNormStrategy = Union[
    RMSNormStrategy, LayerNormStrategy, GroupNormStrategy
]

ConditionalNormStrategy = Union[
    ConditionalRMSNormStrategy,
    ConditionalLayerNormStrategy,
    ConditionalGroupNormStrategy,
]

NormStrategy = Union[UnconditionalNormStrategy, ConditionalNormStrategy]
