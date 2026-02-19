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

"""Tests for dit_blocks."""

from hackable_diffusion.lib.architecture import dit_blocks
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import test_utils
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized


class DiTBlocksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = False

  def test_patch_embedder(self):
    x = jnp.ones((1, 32, 32, 3))
    layer = dit_blocks.PatchEmbedder(patch_size=4, hidden_size=16)
    variables = layer.init(self.key, x)
    # Checking that the variables have expected shapes.
    variables_shapes = test_utils.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'PatchEmbedder_Conv': {
                'kernel': (4, 4, 3, 16),
                'bias': (16,),
            }
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)
    out = layer.apply(variables, x)
    # 32/4 = 8. 8*8 = 64 patches.
    self.assertEqual(out.shape, (1, 64, 16))

  def test_dit_block(self):
    x = jnp.ones((1, 64, 16))
    c = jnp.ones((1, 32))  # Condition
    norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=normalization.NormalizationType.RMS_NORM
    )
    layer = dit_blocks.DiTBlock(
        norm_factory=norm_factory,
        hidden_size=16,
        num_heads=4,
        mlp_ratio=4.0,
    )
    variables = layer.init(
        {'params': self.key, 'dropout': self.key},
        x,
        c,
        is_training=self.is_training,
    )
    # Checking that the variables have expected shapes.
    variables_shapes = test_utils.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'AttnGate': {'bias': (16,), 'kernel': (32, 16)},
            'ConditionalNorm': {
                'Dense_0': {'bias': (32,), 'kernel': (32, 32)},
                'RMSNorm_0': {'scale': (16,)},
            },
            'MLP': {
                'Dense_Hidden_0': {'bias': (64,), 'kernel': (16, 64)},
                'Dense_Output': {'bias': (16,), 'kernel': (64, 16)},
            },
            'MLPGate': {'bias': (16,), 'kernel': (32, 16)},
            'attn': {
                'Dense_K': {'bias': (16,), 'kernel': (16, 16)},
                'Dense_Output': {'bias': (16,), 'kernel': (16, 16)},
                'Dense_Q': {'bias': (16,), 'kernel': (16, 16)},
                'Dense_V': {'bias': (16,), 'kernel': (16, 16)},
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)
    out = layer.apply(variables, x, c, is_training=self.is_training)
    self.assertEqual(out.shape, (1, 64, 16))

  def test_final_layer(self):
    x = jnp.ones((1, 64, 16))
    c = jnp.ones((1, 32))
    norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=normalization.NormalizationType.RMS_NORM
    )
    layer = dit_blocks.FinalLayer(
        norm_factory=norm_factory, patch_size=4, out_channels=3
    )
    variables = layer.init(self.key, x, c)
    # Checking that the variables have expected shapes.
    variables_shapes = test_utils.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'ConditionalNorm': {
                'RMSNorm_0': {
                    'scale': (16,),
                },
                'Dense_0': {
                    'kernel': (32, 32),
                    'bias': (32,),
                },
            },
            'Final_Linear': {
                'kernel': (16, 4 * 4 * 3),
                'bias': (4 * 4 * 3,),
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)
    out = layer.apply(variables, x, c)
    # H = sqrt(64) * 4 = 8 * 4 = 32.
    self.assertEqual(out.shape, (1, 32, 32, 3))


if __name__ == '__main__':
  absltest.main()
