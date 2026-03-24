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

"""Tests for the DiT backbone."""

from hackable_diffusion.lib import test_utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import dit
from hackable_diffusion.lib.architecture import dit_blocks
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Tests
################################################################################


class DiTTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.is_training = True
    self.batch_size, self.h, self.w, self.c = 2, 16, 16, 3
    self.patch_size = (4, 4)
    self.implied_sequence_length = (self.h // self.patch_size[0]) * (
        self.w // self.patch_size[1]
    )
    self.embedding_dim = 32
    self.cond_dim = 17
    self.sequence_length = 33

  def test_output_shape_with_patchify(self):
    data_shape = (self.h, self.w, self.c)
    input_shape = (self.batch_size, *data_shape)
    x = jnp.ones(input_shape)
    model = dit.DiT(
        num_blocks=2,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
        encoder=dit_blocks.Patchify(
            patch_size=self.patch_size, embedding_dim=self.embedding_dim
        ),
        decoder=dit_blocks.DePatchify(
            patch_size=self.patch_size, output_shape=data_shape
        ),
    )
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    output = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, input_shape)

  def test_variable_shapes_with_patchify(self):
    data_shape = (self.h, self.w, self.c)
    input_shape = (self.batch_size, *data_shape)
    x = jnp.ones(input_shape)
    model = dit.DiT(
        num_blocks=2,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
        encoder=dit_blocks.Patchify(
            patch_size=self.patch_size, embedding_dim=self.embedding_dim
        ),
        decoder=dit_blocks.DePatchify(
            patch_size=self.patch_size, output_shape=data_shape
        ),
        absolute_posenc=dit_blocks.PositionalEmbedding(),
    )
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    mlp_hidden = int(self.embedding_dim * 4.0)

    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    variables_shapes = test_utils.get_pytree_shapes(variables)

    block_params = {
        'Dense_Gate_MSA': {
            'kernel': (self.cond_dim, self.embedding_dim),
            'bias': (self.embedding_dim,),
        },
        'Dense_Gate_MLP': {
            'kernel': (self.cond_dim, self.embedding_dim),
            'bias': (self.embedding_dim,),
        },
        'ConditionalNorm': {
            'Dense_0': {
                'kernel': (self.cond_dim, self.embedding_dim * 2),
                'bias': (self.embedding_dim * 2,),
            },
        },
        'MLP': {
            'Dense_Hidden_0': {
                'kernel': (self.embedding_dim, mlp_hidden),
                'bias': (mlp_hidden,),
            },
            'Dense_Output': {
                'kernel': (mlp_hidden, self.embedding_dim),
                'bias': (self.embedding_dim,),
            },
        },
        'attn': {
            'Dense_Q': {
                'kernel': (self.embedding_dim, self.embedding_dim),
                'bias': (self.embedding_dim,),
            },
            'Dense_K': {
                'kernel': (self.embedding_dim, self.embedding_dim),
                'bias': (self.embedding_dim,),
            },
            'Dense_V': {
                'kernel': (self.embedding_dim, self.embedding_dim),
                'bias': (self.embedding_dim,),
            },
            'Dense_Output': {
                'kernel': (self.embedding_dim, self.embedding_dim),
                'bias': (self.embedding_dim,),
            },
            'norm_qk_scale': (1, 1, 1, 1),
        },
    }

    expected_variables_shapes = {
        'params': {
            'encoder': {
                'Dense_Project': {
                    'kernel': (
                        self.patch_size[0] * self.patch_size[1] * self.c,
                        self.embedding_dim,
                    ),
                    'bias': (self.embedding_dim,),
                },
            },
            'absolute_posenc': {
                'PositionalEmbeddingTensor': (
                    1,
                    self.implied_sequence_length,
                    self.embedding_dim,
                ),
            },
            'Block_1': block_params,
            'Block_2': block_params,
            'ConditionalNorm': {
                'Dense_0': {
                    'kernel': (self.cond_dim, self.embedding_dim * 2),
                    'bias': (self.embedding_dim * 2,),
                },
            },
            'decoder': {
                'ConditionalNorm': {
                    'Dense_0': {
                        'kernel': (self.cond_dim, self.embedding_dim * 2),
                        'bias': (self.embedding_dim * 2,),
                    },
                },
                'Dense_Out': {
                    'kernel': (
                        self.embedding_dim,
                        self.patch_size[0] * self.patch_size[1] * self.c,
                    ),
                    'bias': (self.patch_size[0] * self.patch_size[1] * self.c,),
                },
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)

  def test_output_shape_tokens(self):
    input_shape = (self.batch_size, self.sequence_length, self.embedding_dim)
    x = jnp.ones(input_shape)
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    model = dit.DiT(
        num_blocks=2,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
    )
    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    output = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, input_shape)

  def test_missing_adaptive_norm_raises(self):
    x = jnp.ones((self.batch_size, self.sequence_length, self.embedding_dim))
    conditioning_embeddings = {}

    model = dit.DiT(
        num_blocks=1,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
    )
    with self.assertRaises(
        ValueError, msg='adaptive_norm_emb must be provided.'
    ):
      model.init(
          self.key,
          x=x,
          conditioning_embeddings=conditioning_embeddings,
          is_training=self.is_training,
      )

  def test_output_shape_with_cross_attention(self):
    """Verifies output shape when cross-attention conditioning is provided."""
    cross_seq_len = 10
    cross_dim = 24
    input_shape = (self.batch_size, self.sequence_length, self.embedding_dim)
    x = jnp.ones(input_shape)
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
        ConditioningMechanism.CROSS_ATTENTION: jnp.ones(
            (self.batch_size, cross_seq_len, cross_dim)
        ),
    }
    model = dit.DiT(
        num_blocks=2,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
    )
    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    output = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, input_shape)

  def test_output_shape_with_cross_attention_and_mask(self):
    """Verifies output shape when both cross-attention and mask are provided."""
    cross_seq_len = 10
    cross_dim = 24
    input_shape = (self.batch_size, self.sequence_length, self.embedding_dim)
    x = jnp.ones(input_shape)
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
        ConditioningMechanism.CROSS_ATTENTION: jnp.ones(
            (self.batch_size, cross_seq_len, cross_dim)
        ),
        ConditioningMechanism.CROSS_ATTENTION_MASK: jnp.ones(
            (self.batch_size, cross_seq_len), dtype=jnp.bool_
        ),
    }
    model = dit.DiT(
        num_blocks=2,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
    )
    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    output = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, input_shape)

  def test_variable_shapes_with_cross_attention(self):
    """Verifies that cross-attention params are correctly initialized."""
    cross_seq_len = 10
    cross_dim = 24
    input_shape = (self.batch_size, self.sequence_length, self.embedding_dim)
    x = jnp.ones(input_shape)
    conditioning_embeddings = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
        ConditioningMechanism.CROSS_ATTENTION: jnp.ones(
            (self.batch_size, cross_seq_len, cross_dim)
        ),
    }
    model = dit.DiT(
        num_blocks=1,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
    )
    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=self.is_training,
    )
    variables_shapes = test_utils.get_pytree_shapes(variables)
    block_params = variables_shapes['params']['Block_1']

    # Cross-attention gate should be present.
    self.assertIn('Dense_Gate_Cross', block_params)
    self.assertEqual(
        block_params['Dense_Gate_Cross']['kernel'],
        (self.cond_dim, self.embedding_dim),
    )

    # Cross-attention module should be present with correct shapes.
    self.assertIn('cross_attn', block_params)
    cross_attn = block_params['cross_attn']
    # Q projects from embedding_dim
    self.assertEqual(
        cross_attn['Dense_Q']['kernel'],
        (self.embedding_dim, self.embedding_dim),
    )
    # K and V project from cross_dim
    self.assertEqual(
        cross_attn['Dense_K']['kernel'],
        (cross_dim, self.embedding_dim),
    )
    self.assertEqual(
        cross_attn['Dense_V']['kernel'],
        (cross_dim, self.embedding_dim),
    )

  def test_cross_attention_mask_zeros_out_tokens(self):
    """Verifies that masking all cross-attention tokens changes the output."""
    cross_seq_len = 10
    cross_dim = 24
    input_shape = (self.batch_size, self.sequence_length, self.embedding_dim)
    x = jax.random.normal(self.key, input_shape)
    adaptive_norm = jax.random.normal(
        jax.random.PRNGKey(2), (self.batch_size, self.cond_dim)
    )
    cross_emb = jax.random.normal(
        jax.random.PRNGKey(1), (self.batch_size, cross_seq_len, cross_dim)
    )

    model = dit.DiT(
        num_blocks=1,
        block=dit_blocks.DiTBlockAdaLNZero(
            hidden_size=self.embedding_dim, num_heads=4
        ),
    )

    # Init with cross-attention to get correct params.
    conditioning_with_cross = {
        ConditioningMechanism.ADAPTIVE_NORM: adaptive_norm,
        ConditioningMechanism.CROSS_ATTENTION: cross_emb,
        ConditioningMechanism.CROSS_ATTENTION_MASK: jnp.ones(
            (self.batch_size, cross_seq_len), dtype=jnp.bool_
        ),
    }
    variables = model.init(
        self.key,
        x=x,
        conditioning_embeddings=conditioning_with_cross,
        is_training=False,
    )

    # The cross-attention gate (Dense_Gate_Cross) is zero-initialized, so
    # replace it with ones so the gate is active and masking has an effect.
    params = variables['params']
    gate_cross = params['Block_1']['Dense_Gate_Cross']
    gate_cross = jax.tree.map(jnp.ones_like, gate_cross)
    params['Block_1']['Dense_Gate_Cross'] = gate_cross
    variables = {'params': params}

    # Run with all tokens masked out (False = masked).
    conditioning_all_masked = {
        ConditioningMechanism.ADAPTIVE_NORM: adaptive_norm,
        ConditioningMechanism.CROSS_ATTENTION: cross_emb,
        ConditioningMechanism.CROSS_ATTENTION_MASK: jnp.zeros(
            (self.batch_size, cross_seq_len), dtype=jnp.bool_
        ),
    }
    # Run with all tokens unmasked.
    conditioning_all_unmasked = {
        ConditioningMechanism.ADAPTIVE_NORM: adaptive_norm,
        ConditioningMechanism.CROSS_ATTENTION: cross_emb,
        ConditioningMechanism.CROSS_ATTENTION_MASK: jnp.ones(
            (self.batch_size, cross_seq_len), dtype=jnp.bool_
        ),
    }

    output_all_masked = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_all_masked,
        is_training=False,
    )
    output_all_unmasked = model.apply(
        variables,
        x=x,
        conditioning_embeddings=conditioning_all_unmasked,
        is_training=False,
    )

    # The two outputs should differ since masking changes cross-attention.
    self.assertFalse(jnp.allclose(output_all_masked, output_all_unmasked))


if __name__ == '__main__':
  absltest.main()

