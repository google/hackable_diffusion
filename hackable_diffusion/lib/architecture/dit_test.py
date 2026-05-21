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

from hackable_diffusion.lib import test_helpers
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import dit
from hackable_diffusion.lib.architecture import dit_blocks
from hackable_diffusion.lib.architecture import normalization
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

NormalizationType = arch_typing.NormalizationType


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
        block=dit_blocks.DiTBlock(
            hidden_size=self.embedding_dim,
            num_heads=4,
            norm_factory=normalization.NormalizationLayerFactory(
                normalization_method=NormalizationType.RMS_NORM,
                use_conditional_shift=False,
            ),
            use_gates=False,
        ),
        encoder=dit_blocks.Patchify(
            patch_size=self.patch_size, embedding_dim=self.embedding_dim
        ),
        decoder=dit_blocks.DePatchify(
            patch_size=self.patch_size, output_shape=data_shape
        ),
    )
    conditioning_embeddings = {
        'adaptive_norm': jnp.ones(
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
        block=dit_blocks.DiTBlock(
            hidden_size=self.embedding_dim,
            num_heads=4,
            norm_factory=normalization.NormalizationLayerFactory(
                normalization_method=NormalizationType.LAYER_NORM,
                use_bias=False,
                use_scale=False,
            ),
            use_gates=True,
            ffn_type='dense',
            mlp_ratio=4.0,
        ),
        encoder=dit_blocks.Patchify(
            patch_size=self.patch_size, embedding_dim=self.embedding_dim
        ),
        decoder=dit_blocks.DePatchify(
            patch_size=self.patch_size,
            output_shape=data_shape,
        ),
        absolute_posenc=dit_blocks.PositionalEmbedding(),
    )
    conditioning_embeddings = {
        'adaptive_norm': jnp.ones(
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
    variables_shapes = test_helpers.get_pytree_shapes(variables)

    block_params = {
        'Dense_Gate_MSA': {
            'kernel': (self.cond_dim, self.embedding_dim),
            'bias': (self.embedding_dim,),
        },
        'Dense_Gate_MLP': {
            'kernel': (self.cond_dim, self.embedding_dim),
            'bias': (self.embedding_dim,),
        },
        'ConditionalNorm_Attention': {
            'Dense_0': {
                'kernel': (self.cond_dim, self.embedding_dim * 2),
                'bias': (self.embedding_dim * 2,),
            },
        },
        'ffn': {
            'Dense_Up': {
                'kernel': (self.embedding_dim, mlp_hidden),
            },
            'Dense_Down': {
                'kernel': (mlp_hidden, self.embedding_dim),
            },
        },
        'ConditionalNorm_MLP': {
            'Dense_0': {
                'kernel': (self.cond_dim, self.embedding_dim * 2),
                'bias': (self.embedding_dim * 2,),
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
        'adaptive_norm': jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    model = dit.DiT(
        num_blocks=2,
        block=dit_blocks.DiTBlock(
            hidden_size=self.embedding_dim,
            num_heads=4,
            norm_factory=normalization.NormalizationLayerFactory(
                normalization_method=NormalizationType.RMS_NORM,
                use_conditional_shift=False,
            ),
            use_gates=False,
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
        block=dit_blocks.DiTBlock(
            hidden_size=self.embedding_dim,
            num_heads=4,
            norm_factory=normalization.NormalizationLayerFactory(
                normalization_method=NormalizationType.RMS_NORM,
                use_conditional_shift=False,
            ),
            use_gates=False,
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


if __name__ == '__main__':
  absltest.main()
