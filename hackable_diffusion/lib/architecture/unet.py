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

"""Unet with conditional signal."""

from typing import Sequence
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import arch_utils
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import unet_blocks
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp

################################################################################
# MARK: Common types
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

DownsampleType = arch_typing.DownsampleType
UpsampleType = arch_typing.UpsampleType
RoPEPositionType = arch_typing.RoPEPositionType
SkipConnectionMethod = arch_typing.SkipConnectionMethod
ConditionalBackbone = arch_typing.ConditionalBackbone
ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Unet
################################################################################


class Unet(ConditionalBackbone):
  """A U-Net architecture backbone with conditional signals.

  Based on:
  https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py


  Attributes:
    base_channels: Base channel count for the U-Net layers.
    channels_multiplier: Sequence of channel multipliers for each U-Net scale.
    num_residual_blocks: Sequence of residual blocks for each U-Net scale.
    downsample_method: Method for downsampling ('avg_pool' or 'max_pool').
    upsample_method: Method for upsampling ('nearest' or 'bilinear').
    dropout_rate: Sequence of dropout rates for each U-Net scale.
    bottleneck_dropout_rate: Dropout rate for the middle bottleneck.
    self_attention_bool: Sequence indicating if self-attention is used at each
      scale.
    cross_attention_bool: Sequence indicating if cross-attention is used at each
      scale. In the middle blck, cross attention is used by default if
      embeddings for cross attention is given.
    attention_num_heads: Number of attention heads. If set to INVALID_INT, it is
      inferred from head_dim and input channels.
    attention_head_dim: Dimension of each attention head. If set to INVALID_INT,
      it is inferred from num_heads and input channels. One of num_heads or
      head_dim must be INVALID_INT.
    attention_normalize_qk: Whether to normalize query and key in attention as
      in https://arxiv.org/abs/2010.04245.
    attention_use_rope: Whether to use rotary positional embeddings in attention
      as in https://arxiv.org/abs/2104.09864.
    attention_rope_position_type: The type of rotary positional embeddings to
      use.
    normalization_type: Type of normalization to use ('group_norm' or
      'rms_norm').
    normalization_num_groups: Number of groups for GroupNorm, if used.
    activation: Name of the activation function (e.g., 'silu', 'gelu').
    skip_connection_method: Method for skip connections ('unnormalized_add' or
      'normalized_add').
    output_channels: Number of output channels. If None, defaults to the number
      of input channels.
    zero_init_output: If True, the final layer weights are initialized to zero.
    dtype: Data type for model computations.
  """

  # structure
  base_channels: int
  channels_multiplier: Sequence[int]
  num_residual_blocks: Sequence[int]

  # resampling
  downsample_method: DownsampleType
  upsample_method: UpsampleType

  # dropout
  dropout_rate: Sequence[float]
  bottleneck_dropout_rate: float

  # attention
  self_attention_bool: Sequence[bool]
  cross_attention_bool: Sequence[bool]
  attention_num_heads: int
  attention_head_dim: int
  attention_normalize_qk: bool
  attention_use_rope: bool
  attention_rope_position_type: RoPEPositionType

  # normalization
  normalization_type: normalization.NormalizationType
  normalization_num_groups: int | None

  # other
  activation: str
  skip_connection_method: SkipConnectionMethod

  output_channels: int | None = None
  zero_init_output: bool = True
  dtype: DType = jnp.float32

  def setup(self):
    if not (
        len(self.channels_multiplier)
        == len(self.num_residual_blocks)
        == len(self.dropout_rate)
        == len(self.self_attention_bool)
        == len(self.cross_attention_bool)
    ):
      raise ValueError(
          "channels_multiplier, num_residual_blocks, dropout_rate,"
          " self_attention_bool, and cross_attention_bool must have the same"
          f" length, but they are {self.channels_multiplier}, "
          f" {self.num_residual_blocks}, {self.dropout_rate},"
          f" {self.self_attention_bool}, and {self.cross_attention_bool}."
      )

    self.num_scales = len(self.channels_multiplier)
    self.activation_fn = getattr(jax.nn, self.activation)

    self.downsample_fn = arch_utils.get_downsample_fn(self.downsample_method)
    self.upsample_fn = arch_utils.get_upsample_fn(self.upsample_method)

    self.skip_connection_fn = arch_utils.get_skip_connection_fn(
        self.skip_connection_method
    )

    self.norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=self.normalization_type,
        num_groups=self.normalization_num_groups,
        dtype=self.dtype,
    )

    self.post_unconditional_norm = (
        self.norm_factory.unconditional_norm_factory()
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch height width channels"],
      conditioning_embeddings: dict[ConditioningMechanism, Float["batch ..."]],
      is_training: bool,
  ) -> Float["batch height width output_channels"]:

    # Extract conditioning embeddings to use with adaptive normalization.
    adaptive_norm_emb = conditioning_embeddings.get(
        ConditioningMechanism.ADAPTIVE_NORM
    )
    if adaptive_norm_emb is None:
      raise ValueError("adaptive_norm_emb must be provided.")

    # Extract conditioning embeddings to use with cross attention.
    cross_attention_emb = conditioning_embeddings.get(
        ConditioningMechanism.CROSS_ATTENTION
    )
    if any(self.cross_attention_bool) and cross_attention_emb is None:
      raise ValueError(
          "cross_attention_emb must be provided when cross attention is used."
      )
    if cross_attention_emb is not None and cross_attention_emb.ndim == 2:
      cross_attention_emb = cross_attention_emb[:, jnp.newaxis, :]

    # Stack of layers
    num_input_channels = x.shape[-1]
    stack = []

    # Input layer
    x = unet_blocks.InputConvBlock(
        num_output_channels=self.base_channels * self.channels_multiplier[0],
        dtype=self.dtype,
        name="InputBlock",
    )(x)
    stack.append(x)

    # Downsampling
    for i in range(self.num_scales):
      for j in range(self.num_residual_blocks[i]):
        x = unet_blocks.ConvResidualBlock(
            norm_factory=self.norm_factory,
            output_channels=self.base_channels * self.channels_multiplier[i],
            activation_fn=self.activation_fn,
            skip_connection_fn=self.skip_connection_fn,
            dropout_rate=self.dropout_rate[i],
            dtype=self.dtype,
            name=f"Down_{i}_ConvResidualBlock_{j}",
        )(x=x, adaptive_norm_emb=adaptive_norm_emb, is_training=is_training)
        if self.self_attention_bool[i]:
          x = unet_blocks.AttentionResidualBlock(
              norm_factory=self.norm_factory,
              cross_attention_bool=self.cross_attention_bool[i],
              use_rope=self.attention_use_rope,
              rope_position_type=self.attention_rope_position_type,
              skip_connection_fn=self.skip_connection_fn,
              num_heads=self.attention_num_heads,
              head_dim=self.attention_head_dim,
              normalize_qk=self.attention_normalize_qk,
              dtype=self.dtype,
              name=f"Down_{i}_AttentionResidualBlock_{j}",
          )(
              x=x,
              cross_attention_emb=cross_attention_emb,
              is_training=is_training,
          )
        stack.append(x)

      if i != self.num_scales - 1:
        x = unet_blocks.ConvResidualBlock(
            norm_factory=self.norm_factory,
            output_channels=self.base_channels * self.channels_multiplier[i],
            activation_fn=self.activation_fn,
            skip_connection_fn=self.skip_connection_fn,
            resample_type="down",
            downsample_fn=self.downsample_fn,
            dropout_rate=self.dropout_rate[i],
            dtype=self.dtype,
            name=f"Down_{i}_DownsamplingResidualBlock",
        )(x=x, adaptive_norm_emb=adaptive_norm_emb, is_training=is_training)
        stack.append(x)

    # Middle
    middle_ch_multiplier = self.channels_multiplier[-1]

    x = unet_blocks.ConvResidualBlock(
        norm_factory=self.norm_factory,
        output_channels=self.base_channels * middle_ch_multiplier,
        activation_fn=self.activation_fn,
        skip_connection_fn=self.skip_connection_fn,
        dropout_rate=self.bottleneck_dropout_rate,
        dtype=self.dtype,
        name="Middle_PreAttn_ConvResidualBlock",
    )(x=x, adaptive_norm_emb=adaptive_norm_emb, is_training=is_training)
    # uses cross attention by default if cross_attention_emb is given.
    x = unet_blocks.AttentionResidualBlock(
        norm_factory=self.norm_factory,
        cross_attention_bool=True,
        use_rope=self.attention_use_rope,
        rope_position_type=self.attention_rope_position_type,
        skip_connection_fn=self.skip_connection_fn,
        num_heads=self.attention_num_heads,
        head_dim=self.attention_head_dim,
        normalize_qk=self.attention_normalize_qk,
        dtype=self.dtype,
        name="Middle_AttentionResidualBlock",
    )(x=x, cross_attention_emb=cross_attention_emb, is_training=is_training)
    x = unet_blocks.ConvResidualBlock(
        norm_factory=self.norm_factory,
        output_channels=self.base_channels * middle_ch_multiplier,
        activation_fn=self.activation_fn,
        skip_connection_fn=self.skip_connection_fn,
        dropout_rate=self.bottleneck_dropout_rate,
        dtype=self.dtype,
        name="Middle_PostAttn_ConvResidualBlock",
    )(x=x, adaptive_norm_emb=adaptive_norm_emb, is_training=is_training)

    # Upsampling
    for i in reversed(range(self.num_scales)):
      for j in range(self.num_residual_blocks[i] + 1):
        skip = stack.pop()
        x = jnp.concatenate([x, skip], axis=-1)

        x = unet_blocks.ConvResidualBlock(
            norm_factory=self.norm_factory,
            output_channels=self.base_channels * self.channels_multiplier[i],
            activation_fn=self.activation_fn,
            skip_connection_fn=self.skip_connection_fn,
            dropout_rate=self.dropout_rate[i],
            dtype=self.dtype,
            name=f"Up_{i}_ConvResidualBlock_{j}",
        )(x=x, adaptive_norm_emb=adaptive_norm_emb, is_training=is_training)
        if self.self_attention_bool[i]:
          x = unet_blocks.AttentionResidualBlock(
              norm_factory=self.norm_factory,
              cross_attention_bool=self.cross_attention_bool[i],
              use_rope=self.attention_use_rope,
              rope_position_type=self.attention_rope_position_type,
              skip_connection_fn=self.skip_connection_fn,
              num_heads=self.attention_num_heads,
              head_dim=self.attention_head_dim,
              normalize_qk=self.attention_normalize_qk,
              dtype=self.dtype,
              name=f"Up_{i}_AttentionResidualBlock_{j}",
          )(
              x=x,
              cross_attention_emb=cross_attention_emb,
              is_training=is_training,
          )

      if i != 0:
        x = unet_blocks.ConvResidualBlock(
            norm_factory=self.norm_factory,
            output_channels=self.base_channels * self.channels_multiplier[i],
            activation_fn=self.activation_fn,
            skip_connection_fn=self.skip_connection_fn,
            upsample_fn=self.upsample_fn,
            resample_type="up",
            dropout_rate=self.dropout_rate[i],
            dtype=self.dtype,
            name=f"Up_{i}_UpsamplingResidualBlock",
        )(x, adaptive_norm_emb=adaptive_norm_emb, is_training=is_training)

    # Output layer
    num_output_channels = (
        num_input_channels
        if self.output_channels is None
        else self.output_channels
    )

    x = unet_blocks.OutputConvBlock(
        norm_factory=self.norm_factory,
        activation_fn=self.activation_fn,
        num_output_channels=num_output_channels,
        dtype=self.dtype,
        zero_init=self.zero_init_output,
        name="OutputBlock",
    )(x)

    x = utils.optional_bf16_to_fp32(x)
    return x
