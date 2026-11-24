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

"""Unet building blocks."""

import dataclasses
import functools
from typing import Callable, Protocol, TypeAlias
import flax.linen as nn
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import sequence_embedders
import jax
import jax.numpy as jnp
import jaxtyping as jt

################################################################################
# MARK: Type Aliases
################################################################################

DType: TypeAlias = jax_helpers.DType
ActivationFn: TypeAlias = Callable[[jax.Array], jax.Array]
RoPEPositionsFn: TypeAlias = sequence_embedders.RoPEPositionsFn
SquareRoPEPositions = sequence_embedders.SquareRoPEPositions
typechecked = jax_helpers.typechecked


SpatialInput: TypeAlias = jt.Float[jt.Array, "b height width input_channels"]
SpatialOutput: TypeAlias = jt.Float[jt.Array, "b height width output_channels"]
UpsampleOutput: TypeAlias = jt.Float[
    jt.Array, "b height*2 width*2 output_channels"
]
DownsampleOutput: TypeAlias = jt.Float[
    jt.Array, "b height//2 width//2 output_channels"
]

# Reusable NN Components
kernel_init = nn.initializers.lecun_normal()
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3), padding="SAME")
ZerosConv3x3 = functools.partial(
    nn.Conv,
    kernel_size=(3, 3),
    padding="SAME",
    kernel_init=nn.initializers.zeros_init(),
    bias_init=nn.initializers.zeros_init(),
)
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1), padding="SAME")


################################################################################
# MARK: Callable classes for skip connections, downsampling, and upsampling
################################################################################


class SkipConnectionFn(Protocol):
  """Protocol for skip connection functions."""

  def __call__(
      self,
      x: jt.Float[jt.Array, "b height width channels"],
      skip: jt.Float[jt.Array, "b height width channels"],
  ) -> jt.Float[jt.Array, "b height width channels"]:
    ...


class DownsampleFn(Protocol):
  """Protocol for downsample functions."""

  def __call__(
      self, x: jt.Float[jt.Array, "b height width channels"]
  ) -> jt.Float[jt.Array, "b height//2 width//2 channels"]:
    ...


class UpsampleFn(Protocol):
  """Protocol for upsample functions."""

  def __call__(
      self, x: jt.Float[jt.Array, "b height width channels"]
  ) -> jt.Float[jt.Array, "b height*2 width*2 channels"]:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class UnnormalizedAddSkip(SkipConnectionFn):
  """Unnormalized addition skip connection (x + skip)."""

  @typechecked
  def __call__(
      self,
      x: jt.Float[jt.Array, "b height width channels"],
      skip: jt.Float[jt.Array, "b height width channels"],
  ) -> jt.Float[jt.Array, "b height width channels"]:
    return x + skip


@dataclasses.dataclass(frozen=True, kw_only=True)
class NormalizedAddSkip(SkipConnectionFn):
  """Normalized addition skip connection ((x + skip) / sqrt(2))."""

  @typechecked
  def __call__(
      self,
      x: jt.Float[jt.Array, "b height width channels"],
      skip: jt.Float[jt.Array, "b height width channels"],
  ) -> jt.Float[jt.Array, "b height width channels"]:
    return (x + skip) / jnp.sqrt(2)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MaxPoolDownsample(DownsampleFn):
  """Max pooling downsample function."""

  window_shape: tuple[int, int] = (2, 2)
  strides: tuple[int, int] = (2, 2)

  @typechecked
  def __call__(
      self, x: jt.Float[jt.Array, "b height width channels"]
  ) -> jt.Float[jt.Array, "b height//2 width//2 channels"]:
    return nn.max_pool(x, window_shape=self.window_shape, strides=self.strides)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AvgPoolDownsample(DownsampleFn):
  """Average pooling downsample function."""

  window_shape: tuple[int, int] = (2, 2)
  strides: tuple[int, int] = (2, 2)

  @typechecked
  def __call__(
      self, x: jt.Float[jt.Array, "b height width channels"]
  ) -> jt.Float[jt.Array, "b height//2 width//2 channels"]:
    return nn.avg_pool(x, window_shape=self.window_shape, strides=self.strides)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImageResizeUpsample(UpsampleFn):
  """Image resizing upsample function."""

  resize_method: str

  @typechecked
  def __call__(
      self, x: jt.Float[jt.Array, "b height width channels"]
  ) -> jt.Float[jt.Array, "b height*2 width*2 channels"]:
    return jax.image.resize(
        x,
        (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
        method=self.resize_method,
    )


################################################################################
# MARK: Input and Output Blocks
################################################################################


class InputConvBlock(nn.Module):
  """Input embedding layer.

  Applies a 3x3 convolution to the input.

  Attributes:
    num_output_channels: The number of output channels.
    dtype: The data type of the computation.
  """

  num_output_channels: int
  dtype: DType = jnp.float32

  @nn.compact
  @typechecked
  def __call__(self, x: SpatialInput) -> SpatialOutput:
    x = Conv3x3(
        padding="SAME",
        features=self.num_output_channels,
        dtype=self.dtype,
    )(x)
    return x


class OutputConvBlock(nn.Module):
  """Output projection layer.

  Performs the following operations:
  Normalization -> Activation -> 3x3 Convolution.

  Attributes:
    num_output_channels: The number of output channels.
    norm_strategy: Strategy for building normalization layers.
    activation_fn: The activation function.
    zero_init: Whether to initialize the output convolution with zeros.
    dtype: The data type of the computation.
  """

  num_output_channels: int
  norm_strategy: normalization.NormStrategy
  activation_fn: ActivationFn
  zero_init: bool
  dtype: DType = jnp.float32

  def setup(self):
    self.norm = self.norm_strategy.build_layer(name="Norm")

    if self.zero_init:
      self.output_conv = ZerosConv3x3
    else:
      self.output_conv = Conv3x3  # default kernel init

  @nn.compact
  @typechecked
  def __call__(self, x: SpatialInput) -> SpatialOutput:
    """Projects the output tensor."""

    x = self.norm(x)
    x = self.activation_fn(x)

    x = self.output_conv(
        features=self.num_output_channels,
        dtype=self.dtype,
    )(x)
    return x


################################################################################
# MARK: Residual Block With Optional Resampling
################################################################################


class ConvResidualBlock(nn.Module):
  """Convolutional residual block with optional resampling.

  Attributes:
    norm_strategy: Strategy for building normalization layers.
    output_channels: The number of output channels.
    activation_fn: The activation function.
    skip_connection_fn: The skip connection function.
    resample_type: The type of resampling to apply ('down', 'up', or None).
    downsample_fn: The downsampling function to use if resample_type is 'down'.
    upsample_fn: The upsampling function to use if resample_type is 'up'.
    dropout_rate: The dropout rate.
    dtype: The data type of the computation.
  """

  uncond_norm_strategy: normalization.NormStrategy
  cond_norm_strategy: normalization.NormStrategy
  output_channels: int
  activation_fn: ActivationFn
  skip_connection_fn: SkipConnectionFn
  resample_fn: DownsampleFn | UpsampleFn | None = None
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32

  def setup(self):
    self.norm = self.uncond_norm_strategy.build_layer(name="Norm")
    self.adaptive_norm = self.cond_norm_strategy.build_layer(
        name="AdaptiveNorm"
    )

    self.init_input = kernel_init
    self.init_output = nn.initializers.zeros_init()

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: SpatialInput,
      adaptive_norm_emb: jt.Float[jt.Array, "b emb_dim"],
      is_training: bool,
  ) -> SpatialOutput | UpsampleOutput | DownsampleOutput:
    input_channels = x.shape[-1]
    skip = x
    x = self.norm(x)
    x = self.activation_fn(x)

    if self.resample_fn:
      x = self.resample_fn(x)
      skip = self.resample_fn(skip)

    x = Conv3x3(
        features=self.output_channels,
        kernel_init=self.init_input,
        dtype=self.dtype,
    )(x)

    x = self.adaptive_norm(x, self.activation_fn(adaptive_norm_emb))
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(x)
    x = Conv3x3(
        features=self.output_channels,
        kernel_init=self.init_output,
        dtype=self.dtype,
    )(x)

    if self.output_channels != input_channels:
      skip = Conv1x1(features=self.output_channels, dtype=self.dtype)(skip)

    x = self.skip_connection_fn(x, skip)

    return x


################################################################################
# MARK: Attention Residual Block
################################################################################


class AttentionResidualBlock(nn.Module):
  """Attention residual block.

  Performs the following operations:
  Normalization -> Self-Attention (or Cross-Attention) -> Add skip connection.


  Attributes:
    norm_strategy: Strategy for building normalization layers.
    cross_attention_bool: If True, uses cross-attention with
      `cross_attention_emb` as key/value source if `cross_attention_emb` is not
      None. If False, uses self-attention.
    use_rope: Whether to use rotary positional embeddings in attention.
    rope_positions_fn: The position function of rotary positional embeddings.
    skip_connection_fn: The skip connection function.
    attention_heads_spec: num_heads and head_dim for the attention mecanism.
    normalize_qk: Whether to normalize query and key in attention.
    dropout_rate: Dropout rate applied to attention weights.
    dtype: The data type of the computation.
  """

  norm_strategy: normalization.NormStrategy
  cross_attention_bool: bool
  use_rope: bool
  rope_positions_fn: RoPEPositionsFn
  skip_connection_fn: SkipConnectionFn
  attention_heads_spec: attention.AttentionHeadsSpec
  normalize_qk: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32

  def setup(self):
    self.norm = self.norm_strategy.build_layer(name="Norm")

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: jt.Float[jt.Array, "b height width channels"],
      cross_attention_emb: jt.Float[jt.Array, "b seq cond_dim2"] | None,
      *,
      is_training: bool,
  ) -> jt.Float[jt.Array, "b height width channels"]:
    skip = x
    b, h, w, channels = x.shape
    x = self.norm(x)
    x = x.reshape(b, h * w, channels)
    x = attention.MultiHeadAttention(
        attention_heads_spec=self.attention_heads_spec,
        use_rope=self.use_rope,
        normalize_qk=self.normalize_qk,
        rope_positions_fn=self.rope_positions_fn,
        zero_init_output=True,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
    )(
        x=x,
        c=cross_attention_emb if self.cross_attention_bool else None,
        is_training=is_training,
    )
    x = x.reshape(b, h, w, channels)
    x = self.skip_connection_fn(x, skip)
    return x
