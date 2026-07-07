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

"""Tests for unet."""

import dataclasses

from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import sequence_embedders
from hackable_diffusion.lib.architecture import unet
from hackable_diffusion.lib.architecture import unet_blocks
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

RoPEPositionsFn = sequence_embedders.RoPEPositionsFn
SquareRoPEPositions = sequence_embedders.SquareRoPEPositions


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
  downsample_fn: unet_blocks.DownsampleFn = unet_blocks.AvgPoolDownsample()
  upsample_fn: unet_blocks.UpsampleFn = unet_blocks.ImageResizeUpsample(
      resize_method='nearest'
  )

  # dropout
  dropout_rate: tuple[float, ...] = (0.0, 0.0)
  bottleneck_dropout_rate: float = 0.0

  # attention
  self_attention_bool: tuple[bool, ...] = (False, True)
  cross_attention_bool: tuple[bool, ...] = (False, True)
  attention_heads_spec: attention.AttentionHeadsSpec = attention.AttentionHeadsSpec(
      head_dim=16
  )
  attention_normalize_qk: bool = True
  attention_use_rope: bool = False
  attention_rope_positions_fn: RoPEPositionsFn = SquareRoPEPositions()

  # normalization
  uncond_norm_strategy: normalization.NormStrategy = (
      normalization.GroupNormStrategy(num_groups=4)
  )
  cond_norm_strategy: normalization.ConditionalNormStrategy = (
      normalization.ConditionalGroupNormStrategy(
          num_groups=4,
          use_shift=True,
      )
  )

  # other
  activation: str = 'silu'
  skip_connection_fn: unet_blocks.SkipConnectionFn = (
      unet_blocks.UnnormalizedAddSkip()
  )

  output_channels: int | None = None
  zero_init_output: bool = False


DEFAULT_CONFIG = Config()
RMSNORM_CONFIG = Config(
    uncond_norm_strategy=normalization.RMSNormStrategy(),
    cond_norm_strategy=normalization.ConditionalRMSNormStrategy(
        use_shift=True
    ),
)
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
        'adaptive_norm': jnp.ones((2, 32)),
        'cross_attention': jnp.ones((2, 16, 32)),
    }
    x = jnp.ones(x_shape)
    model = unet.Unet(**config.__dict__, dtype=jnp.float32)  # pyrefly: ignore[bad-argument-type]
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
    self.assertEqual(output.shape, x_shape)  # pyrefly: ignore[missing-attribute]

  @parameterized.named_parameters(
      ('output_channels', OUTPUT_CHANNELS_CONFIG),
  )
  def test_output_num_channels(self, config: Config):
    """Tests Unet output block name."""
    num_input_channels = 3
    x_shape = (2, 16, 16, num_input_channels)
    conditioning_embeddings = {
        'adaptive_norm': jnp.ones((2, 32)),
        'cross_attention': jnp.ones((2, 16, 32)),
    }
    x = jnp.ones(x_shape)
    model = unet.Unet(
        **config.__dict__,  # pyrefly: ignore[bad-argument-type]
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
    self.assertEqual(output.shape, x_shape[:-1] + (config.output_channels,))  # pyrefly: ignore[missing-attribute]


if __name__ == '__main__':
  absltest.main()
