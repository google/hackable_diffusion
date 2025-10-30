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

"""Unet building blocks."""

from typing import Literal
import flax.linen as nn
from hackable_diffusion.lib import array_annotations
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import arch_utils
from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp

################################################################################
# MARK: Common types and aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

ActivationFn = arch_typing.ActivationFn
SkipConnectionFn = arch_typing.SkipConnectionFn
UpsampleFn = arch_typing.UpsampleFn
DownsampleFn = arch_typing.DownsampleFn
RoPEPositionType = arch_typing.RoPEPositionType
NormalizationLayerFactory = normalization.NormalizationLayerFactory

kernel_init = arch_utils.kernel_init
Conv3x3 = arch_utils.Conv3x3
ZerosConv3x3 = arch_utils.ZerosConv3x3
Conv1x1 = arch_utils.Conv1x1

BaseInput = Float["batch height width input_channels"]
BaseOutput = Float["batch height width output_channels"]
UpsampleOutput = Float["batch height*2 width*2 output_channels"]
DownsampleOutput = Float["batch height/2 width/2 output_channels"]

################################################################################
# MARK: Input and Output blocks
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
  def __call__(self, x: BaseInput) -> BaseOutput:
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
    norm_factory: Factory for creating normalization layers.
    activation_fn: The activation function.
    zero_init: Whether to initialize the output convolution with zeros.
    dtype: The data type of the computation.
  """

  num_output_channels: int
  norm_factory: NormalizationLayerFactory
  activation_fn: ActivationFn
  zero_init: bool
  dtype: DType = jnp.float32

  def setup(self):
    self.unconditional_norm = self.norm_factory.unconditional_norm_factory()

    if self.zero_init:
      self.output_conv = ZerosConv3x3
    else:
      self.output_conv = Conv3x3  # default kernel init

  @nn.compact
  @typechecked
  def __call__(self, x: BaseInput) -> BaseOutput:
    """Projects the output tensor."""

    x = self.unconditional_norm(x)
    x = self.activation_fn(x)

    x = self.output_conv(
        features=self.num_output_channels,
        dtype=self.dtype,
    )(x)
    return x


################################################################################
# MARK: Residual block with optional resampling
################################################################################


class ConvResidualBlock(nn.Module):
  """Convolutional residual block with optional resampling.

  Attributes:
    norm_factory: Factory for creating normalization layers.
    output_channels: The number of output channels.
    activation_fn: The activation function.
    skip_connection_fn: The skip connection function.
    resample_type: The type of resampling to apply ('down', 'up', or None).
    downsample_fn: The downsampling function to use if resample_type is 'down'.
    upsample_fn: The upsampling function to use if resample_type is 'up'.
    dropout_rate: The dropout rate.
    dtype: The data type of the computation.
  """

  norm_factory: NormalizationLayerFactory
  output_channels: int
  activation_fn: ActivationFn
  skip_connection_fn: SkipConnectionFn
  resample_type: Literal["down", "up"] | None = None
  downsample_fn: DownsampleFn | None = None
  upsample_fn: UpsampleFn | None = None
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32

  def setup(self):
    if self.resample_type == "down" and self.downsample_fn is None:
      raise ValueError("downsample_fn must be provided for down-resampling.")

    if self.resample_type == "up" and self.upsample_fn is None:
      raise ValueError("upsample_fn must be provided for up-resampling.")

    self.unconditional_norm = self.norm_factory.unconditional_norm_factory()
    self.conditional_norm = self.norm_factory.conditional_norm_factory()

    self.init_input = kernel_init
    self.init_output = nn.initializers.zeros_init()

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch height width channels"],
      adaptive_norm_emb: Float["batch emb_dim"],
      is_training: bool,
  ) -> BaseOutput | UpsampleOutput | DownsampleOutput:
    input_channels = x.shape[-1]
    skip = x
    x = self.unconditional_norm(x)
    x = self.activation_fn(x)

    if self.resample_type is not None:
      if self.resample_type == "down":
        x = self.downsample_fn(x)
        skip = self.downsample_fn(skip)
      elif self.resample_type == "up":
        x = self.upsample_fn(x)
        skip = self.upsample_fn(skip)
      else:
        raise ValueError(f"Unknown resample type: {self.resample_type}")

    x = Conv3x3(
        features=self.output_channels,
        kernel_init=self.init_input,
        dtype=self.dtype,
    )(x)

    x = self.conditional_norm(x, self.activation_fn(adaptive_norm_emb))
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

    # check types
    if self.downsample_fn is not None:
      array_annotations.check_type(x, DownsampleOutput)
    elif self.upsample_fn is not None:
      array_annotations.check_type(x, UpsampleOutput)
    else:
      array_annotations.check_type(x, BaseOutput)

    return x


################################################################################
# MARK: Attention residual block
################################################################################


class AttentionResidualBlock(nn.Module):
  """Attention residual block.

  Performs the following operations:
  Normalization -> Self-Attention (or Cross-Attention) -> Add skip connection.


  Attributes:
    norm_factory: Factory for creating normalization layers.
    cross_attention_bool: If True, uses cross-attention with
      `cross_attention_emb` as key/value source if `cross_attention_emb` is not
      None. If False, uses self-attention.
    use_rope: Whether to use rotary positional embeddings in attention.
    rope_position_type: The type of rotary positional embeddings to use.
    skip_connection_fn: The skip connection function.
    num_heads: The number of attention heads. If set to INVALID_INT, it is
      inferred from head_dim and input channels.
    head_dim: The dimension of each attention head. If set to INVALID_INT, it is
      inferred from num_heads and input channels. One of num_heads or head_dim
      must be INVALID_INT.
    normalize_qk: Whether to normalize query and key in attention.
    dtype: The data type of the computation.
  """

  norm_factory: NormalizationLayerFactory
  cross_attention_bool: bool
  use_rope: bool
  rope_position_type: RoPEPositionType
  skip_connection_fn: SkipConnectionFn
  num_heads: int
  head_dim: int
  normalize_qk: bool = False
  dtype: DType = jnp.float32

  def setup(self):
    self.unconditional_norm = self.norm_factory.unconditional_norm_factory()

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch height width channels"],
      cross_attention_emb: Float["batch seq cond_dim2"] | None,
      is_training: bool,
  ) -> Float["batch height width channels"]:
    skip = x
    b, h, w, channels = x.shape
    x = self.unconditional_norm(x)
    x = x.reshape(b, h * w, channels)
    x = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        use_rope=self.use_rope,
        normalize_qk=self.normalize_qk,
        rope_position_type=self.rope_position_type,
        zero_init_output=True,
        dtype=self.dtype,
    )(x=x, c=cross_attention_emb if self.cross_attention_bool else None)
    x = x.reshape(b, h, w, channels)
    x = self.skip_connection_fn(x, skip)
    return x
