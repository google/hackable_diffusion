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

"""Tests for dit."""

from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import dit
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import test_utils
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized


class DiTTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = False

  def test_dit(self):
    x = jnp.ones((1, 32, 32, 3))
    c = jnp.ones((1, 32))
    cond_embeddings = {arch_typing.ConditioningMechanism.ADAPTIVE_NORM: c}

    model = dit.DiT(
        patch_size=4,
        hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        normalization_type=normalization.NormalizationType.RMS_NORM,
    )

    variables = model.init(
        {'params': self.key, 'dropout': self.key},
        x,
        cond_embeddings,
        is_training=self.is_training,
    )
    variables_shapes = test_utils.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'AdditiveSequenceEmbedding_0': {
                'PositionalEmbeddingTensor': (1, 64, 16)
            },
            'DiTBlock_0': {
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
            },
            'DiTBlock_1': {
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
            },
            'FinalLayer': {
                'ConditionalNorm': {
                    'Dense_0': {'bias': (32,), 'kernel': (32, 32)},
                    'RMSNorm_0': {'scale': (16,)},
                },
                'Final_Linear': {'bias': (48,), 'kernel': (16, 48)},
            },
            'patch_embedder': {
                'PatchEmbedder_Conv': {'bias': (16,), 'kernel': (4, 4, 3, 16)}
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)
    out = model.apply(
        variables, x, cond_embeddings, is_training=self.is_training
    )
    self.assertEqual(out.shape, (1, 32, 32, 3))


if __name__ == '__main__':
  absltest.main()
