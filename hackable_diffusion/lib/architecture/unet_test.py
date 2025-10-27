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

"""Tests for unet."""

import dataclasses

from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import unet
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
INVALID_INT = arch_typing.INVALID_INT
ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Tests
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
  # structure
  base_channels: int = 16
  channels_multiplier: tuple[int, ...] = (1, 2)
  num_residual_blocks: tuple[int, ...] = (1, 1)

  # resampling
  downsample_method: DownsampleType = DownsampleType.AVG_POOL
  upsample_method: UpsampleType = UpsampleType.NEAREST

  # dropout
  dropout_rate: tuple[float, ...] = (0.0, 0.0)
  bottleneck_dropout_rate: float = 0.0

  # attention
  self_attention_bool: tuple[bool, ...] = (False, True)
  cross_attention_bool: tuple[bool, ...] = (False, True)
  attention_num_heads: int = INVALID_INT
  attention_head_dim: int = 16
  attention_normalize_qk: bool = True
  attention_use_rope: bool = False
  attention_rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE

  # normalization
  normalization_type: NormalizationType = NormalizationType.GROUP_NORM
  normalization_num_groups: int = 4

  # other
  activation: str = 'silu'
  skip_connection_method: SkipConnectionMethod = (
      SkipConnectionMethod.UNNORMALIZED_ADD
  )

  output_channels: int | None = None
  zero_init_output: bool = False


DEFAULT_CONFIG = Config()
RMSNORM_CONFIG = Config(normalization_type=NormalizationType.RMS_NORM)
OUTPUT_CHANNELS_CONFIG = Config(output_channels=2)


class UnetTest(parameterized.TestCase):
  """Tests for Unet."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = True

  # MARK: Unet tests

  @parameterized.named_parameters(
      ('default', DEFAULT_CONFIG),
      ('rms_norm', RMSNORM_CONFIG),
  )
  def test_output_shape(self, config: Config):
    """Tests Unet output shape."""
    x_shape = (2, 16, 16, 3)
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones((2, 32)),
        ConditioningMechanism.CROSS_ATTENTION: jnp.ones((2, 16, 32)),
    }
    x = jnp.ones(x_shape)
    model = unet.Unet(**dataclasses.asdict(config), dtype=jnp.float32)
    variables = model.init(
        {'params': self.key, 'dropout': self.key},
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    output = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
        rngs={'dropout': self.key},
    )
    self.assertEqual(output.shape, x_shape)

  @parameterized.named_parameters(
      ('output_channels', OUTPUT_CHANNELS_CONFIG),
  )
  def test_output_num_channels(self, config: Config):
    """Tests Unet output block name."""
    num_input_channels = 3
    x_shape = (2, 16, 16, num_input_channels)
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones((2, 32)),
        ConditioningMechanism.CROSS_ATTENTION: jnp.ones((2, 16, 32)),
    }
    x = jnp.ones(x_shape)
    model = unet.Unet(
        **dataclasses.asdict(config),
        dtype=jnp.float32,
    )
    variables = model.init(
        {'params': self.key, 'dropout': self.key},
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )

    output = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
        rngs={'dropout': self.key},
    )
    self.assertEqual(output.shape, x_shape[:-1] + (config.output_channels,))


if __name__ == '__main__':
  absltest.main()
