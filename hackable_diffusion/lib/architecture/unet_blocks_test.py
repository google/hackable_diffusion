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

"""Tests for unet_blocks."""

from typing import Literal, Tuple

from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import arch_utils
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import unet_blocks
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

RoPEPositionType = arch_typing.RoPEPositionType
NormalizationType = arch_typing.NormalizationType
DownsampleType = arch_typing.DownsampleType
UpsampleType = arch_typing.UpsampleType
SkipConnectionMethod = arch_typing.SkipConnectionMethod
ResampleType = Literal['down', 'up'] | None
INVALID_INT = arch_typing.INVALID_INT

################################################################################
# MARK: Tests
################################################################################


class InputBlockTest(parameterized.TestCase):
  """Tests for InputBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)

  def test_input_block_output_shape(self):
    """Tests InputBlock output shape."""
    num_output_channels = 16
    block = unet_blocks.InputConvBlock(
        num_output_channels=num_output_channels, dtype=jnp.float32
    )
    x = jnp.ones((2, 16, 16, 3))
    variables = block.init(self.key, x)
    output = block.apply(variables, x)
    self.assertEqual(output.shape, (2, 16, 16, num_output_channels))


class OutputBlockTest(parameterized.TestCase):
  """Tests for OutputBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)

  @parameterized.named_parameters(
      ('group_norm_zero_init', 'group_norm', True),
      ('rms_norm_zero_init', 'rms_norm', True),
      ('group_norm', 'group_norm', False),
      ('rms_norm', 'rms_norm', False),
  )
  def test_output_block_output_shape(
      self, normalization_type: NormalizationType, zero_init: bool
  ):
    """Tests OutputBlock output shape."""
    num_output_channels = 3
    norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=normalization_type,
        num_groups=4 if normalization_type == 'group_norm' else None,
        dtype=jnp.float32,
    )
    block = unet_blocks.OutputConvBlock(
        num_output_channels=num_output_channels,
        norm_factory=norm_factory,
        activation_fn=jax.nn.silu,
        zero_init=zero_init,
        dtype=jnp.float32,
    )
    x = jnp.ones((2, 16, 16, 16))
    variables = block.init(self.key, x)
    output = block.apply(variables, x)
    self.assertEqual(output.shape, (2, 16, 16, num_output_channels))


class ConvResidualBlockTest(parameterized.TestCase):
  """Tests for ConvResidualBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = True

  def _get_conv_residual_block(
      self,
      resample_type: ResampleType,
      normalization_type: NormalizationType,
  ) -> unet_blocks.ConvResidualBlock:
    """Returns a ConvResidualBlock for testing."""
    norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=normalization_type,
        num_groups=4 if normalization_type == 'group_norm' else None,
        dtype=jnp.float32,
    )
    skip_connection_fn = arch_utils.get_skip_connection_fn(
        SkipConnectionMethod.UNNORMALIZED_ADD
    )
    downsample_fn = arch_utils.get_downsample_fn(DownsampleType.AVG_POOL)
    upsample_fn = arch_utils.get_upsample_fn(UpsampleType.NEAREST)

    return unet_blocks.ConvResidualBlock(
        norm_factory=norm_factory,
        output_channels=32,
        activation_fn=jax.nn.silu,
        skip_connection_fn=skip_connection_fn,
        downsample_fn=downsample_fn if resample_type == 'down' else None,
        upsample_fn=upsample_fn if resample_type == 'up' else None,
        dropout_rate=0.1,
        resample_type=resample_type,
        dtype=jnp.float32,
    )

  @parameterized.named_parameters(
      ('downsample_group_norm', 'down', 'group_norm', (2, 8, 8, 32)),
      ('upsample_group_norm', 'up', 'group_norm', (2, 32, 32, 32)),
      ('same_group_norm', None, 'group_norm', (2, 16, 16, 32)),
      ('downsample_rms_norm', 'down', 'rms_norm', (2, 8, 8, 32)),
      ('upsample_rms_norm', 'up', 'rms_norm', (2, 32, 32, 32)),
      ('same_rms_norm', None, 'rms_norm', (2, 16, 16, 32)),
  )
  def test_conv_residual_block_output_shape(
      self,
      resample_type: ResampleType,
      normalization_type: NormalizationType,
      expected_shape: Tuple[int, ...],
  ):
    """Tests ConvResidualBlock output shape."""
    block = self._get_conv_residual_block(
        resample_type=resample_type,
        normalization_type=normalization_type,
    )
    x = jnp.ones((2, 16, 16, 16))
    adaptive_norm_emb = jnp.ones((2, 32))
    variables = block.init(
        {'params': self.key, 'dropout': self.key},
        x=x,
        adaptive_norm_emb=adaptive_norm_emb,
        is_training=self.is_training,
    )
    output = block.apply(
        variables,
        x=x,
        adaptive_norm_emb=adaptive_norm_emb,
        is_training=self.is_training,
        rngs={'dropout': self.key},
    )
    self.assertEqual(output.shape, expected_shape)


class AttentionResidualBlockTest(parameterized.TestCase):
  """Tests for AttentionResidualBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = True

  def _get_attention_residual_block(
      self,
      cross_attention_bool: bool,
      normalization_type: NormalizationType,
  ) -> unet_blocks.AttentionResidualBlock:
    """Returns an AttentionResidualBlock for testing."""
    norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=normalization_type,
        num_groups=4 if normalization_type == 'group_norm' else None,
        dtype=jnp.float32,
    )
    skip_connection_fn = arch_utils.get_skip_connection_fn(
        SkipConnectionMethod.UNNORMALIZED_ADD
    )
    return unet_blocks.AttentionResidualBlock(
        norm_factory=norm_factory,
        skip_connection_fn=skip_connection_fn,
        cross_attention_bool=cross_attention_bool,
        dtype=jnp.float32,
        head_dim=16,
        num_heads=INVALID_INT,
        normalize_qk=True,
        use_rope=False,
        rope_position_type=RoPEPositionType.SQUARE,
    )

  @parameterized.named_parameters(
      ('self_attn_group_norm', False, 'group_norm'),
      ('cross_attn_group_norm', True, 'group_norm'),
      ('self_attn_rms_norm', False, 'rms_norm'),
      ('cross_attn_rms_norm', True, 'rms_norm'),
  )
  def test_attention_residual_block_output_shape(
      self,
      cross_attention_bool: bool,
      normalization_type: NormalizationType,
  ):
    """Tests AttentionResidualBlock output shape."""
    block = self._get_attention_residual_block(
        cross_attention_bool=cross_attention_bool,
        normalization_type=normalization_type,
    )
    x_shape = (2, 16, 16, 32)
    x = jnp.ones(x_shape)
    cross_attention_emb = jnp.ones((2, 3, 8))
    variables = block.init(
        {'params': self.key, 'dropout': self.key},
        x=x,
        cross_attention_emb=cross_attention_emb,
        is_training=self.is_training,
    )
    output = block.apply(
        variables,
        x=x,
        cross_attention_emb=cross_attention_emb,
        is_training=self.is_training,
        rngs={'dropout': self.key},
    )
    self.assertEqual(output.shape, x_shape)


if __name__ == '__main__':
  absltest.main()
