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

"""Tests for unet_blocks."""

from typing import Literal, Tuple


from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import sequence_embedders
from hackable_diffusion.lib.architecture import unet_blocks
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

SquareRoPEPositions = sequence_embedders.SquareRoPEPositions
ResampleType = Literal['down', 'up'] | None


def _get_norm_strategy(
    normalization_type: str,
) -> tuple[normalization.NormStrategy, normalization.ConditionalNormStrategy]:
  if normalization_type == 'default_group_norm':
    uncond = normalization.GroupNormStrategy(num_groups=4)
    cond = normalization.ConditionalGroupNormStrategy(
        num_groups=4, use_shift=True
    )
  elif normalization_type == 'default_rms_norm':
    uncond = normalization.RMSNormStrategy()
    cond = normalization.ConditionalRMSNormStrategy(use_shift=True)
  elif normalization_type == 'default_layer_norm':
    uncond = normalization.LayerNormStrategy()
    cond = normalization.ConditionalLayerNormStrategy(use_shift=True)
  else:
    raise ValueError(f'Unknown normalization type {normalization_type}')

  return uncond, cond


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
    self.assertEqual(output.shape, (2, 16, 16, num_output_channels))  # pyrefly: ignore[missing-attribute]


class OutputBlockTest(parameterized.TestCase):
  """Tests for OutputBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)

  @parameterized.named_parameters(
      ('group_norm_zero_init', 'default_group_norm', True),
      ('rms_norm_zero_init', 'default_rms_norm', True),
      ('group_norm', 'default_group_norm', False),
      ('rms_norm', 'default_rms_norm', False),
  )
  def test_output_block_output_shape(
      self, normalization_type: str, zero_init: bool
  ):
    """Tests OutputBlock output shape."""
    num_output_channels = 3
    uncond_norm_strategy, _ = _get_norm_strategy(normalization_type)
    block = unet_blocks.OutputConvBlock(
        num_output_channels=num_output_channels,
        norm_strategy=uncond_norm_strategy,
        activation_fn=jax.nn.silu,
        zero_init=zero_init,
        dtype=jnp.float32,
    )
    x = jnp.ones((2, 16, 16, 16))
    variables = block.init(self.key, x)
    output = block.apply(variables, x)
    self.assertEqual(output.shape, (2, 16, 16, num_output_channels))  # pyrefly: ignore[missing-attribute]


class ConvResidualBlockTest(parameterized.TestCase):
  """Tests for ConvResidualBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = True

  def _get_conv_residual_block(
      self,
      resample_type: ResampleType,
      normalization_type: str,
  ) -> unet_blocks.ConvResidualBlock:
    uncond_norm_strategy, cond_norm_strategy = _get_norm_strategy(
        normalization_type
    )

    downsample_fn = unet_blocks.AvgPoolDownsample()
    upsample_fn = unet_blocks.ImageResizeUpsample(resize_method='nearest')

    return unet_blocks.ConvResidualBlock(
        uncond_norm_strategy=uncond_norm_strategy,
        cond_norm_strategy=cond_norm_strategy,
        output_channels=32,
        activation_fn=jax.nn.silu,
        skip_connection_fn=unet_blocks.UnnormalizedAddSkip(),
        resample_fn=(
            downsample_fn
            if resample_type == 'down'
            else (upsample_fn if resample_type == 'up' else None)
        ),
        dropout_rate=0.1,
        dtype=jnp.float32,
    )

  @parameterized.named_parameters(
      ('downsample_group_norm', 'down', 'default_group_norm', (2, 8, 8, 32)),
      ('upsample_group_norm', 'up', 'default_group_norm', (2, 32, 32, 32)),
      ('same_group_norm', None, 'default_group_norm', (2, 16, 16, 32)),
      ('downsample_rms_norm', 'down', 'default_rms_norm', (2, 8, 8, 32)),
      ('upsample_rms_norm', 'up', 'default_rms_norm', (2, 32, 32, 32)),
      ('same_rms_norm', None, 'default_rms_norm', (2, 16, 16, 32)),
  )
  def test_conv_residual_block_output_shape(
      self,
      resample_type: ResampleType,
      normalization_type: str,
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
    self.assertEqual(output.shape, expected_shape)  # pyrefly: ignore[missing-attribute]


class AttentionResidualBlockTest(parameterized.TestCase):
  """Tests for AttentionResidualBlock."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = True

  def _get_attention_residual_block(
      self,
      cross_attention_bool: bool,
      normalization_type: str,
  ) -> unet_blocks.AttentionResidualBlock:
    """Returns an AttentionResidualBlock for testing."""
    uncond_norm_strategy, _ = _get_norm_strategy(normalization_type)
    return unet_blocks.AttentionResidualBlock(
        norm_strategy=uncond_norm_strategy,
        skip_connection_fn=unet_blocks.UnnormalizedAddSkip(),
        cross_attention_bool=cross_attention_bool,
        dtype=jnp.float32,
        attention_heads_spec=attention.AttentionHeadsSpec(head_dim=16),
        normalize_qk=True,
        use_rope=False,
        rope_positions_fn=SquareRoPEPositions(),
    )

  @parameterized.named_parameters(
      ('self_attention_group_norm', False, 'default_group_norm'),
      ('cross_attention_group_norm', True, 'default_group_norm'),
      ('self_attention_rms_norm', False, 'default_rms_norm'),
      ('cross_attention_rms_norm', True, 'default_rms_norm'),
  )
  def test_attention_residual_block_output_shape(
      self,
      cross_attention_bool: bool,
      normalization_type: str,
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
    self.assertEqual(output.shape, x_shape)  # pyrefly: ignore[missing-attribute]


if __name__ == '__main__':
  absltest.main()
