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

"""Tests for the DiT blocks."""

from hackable_diffusion.lib import test_helpers
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import dit_blocks
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

INVALID_INT = arch_typing.INVALID_INT


class DiTBlockAdaLNZeroTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)

    self.batch, self.n, self.d, self.c = 2, 16, 32, 64

  @parameterized.named_parameters(
      ('num_heads', 4, INVALID_INT),
      ('head_dim', INVALID_INT, 8),
  )
  def test_output_shape(self, num_heads, head_dim):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    module = dit_blocks.DiTBlockAdaLNZero(
        hidden_size=self.d,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=4.0,
    )
    variables = module.init(self.key, x, cond, is_training=False)
    output = module.apply(variables, x, cond, is_training=False)
    self.assertEqual(output.shape, input_shape)

  def test_variable_shapes(self):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    mlp_hidden = int(self.d * 4.0)
    module = dit_blocks.DiTBlockAdaLNZero(
        hidden_size=self.d, num_heads=4, mlp_ratio=4.0
    )
    variables = module.init(self.key, x, cond, is_training=False)
    variables_shapes = test_helpers.get_pytree_shapes(variables)

    expected_variables_shapes = {
        'params': {
            'Dense_Gate_MSA': {
                'kernel': (self.c, self.d),
                'bias': (self.d,),
            },
            'Dense_Gate_MLP': {
                'kernel': (self.c, self.d),
                'bias': (self.d,),
            },
            'ConditionalNorm_Attention': {
                'Dense_0': {
                    'kernel': (self.c, self.d * 2),
                    'bias': (self.d * 2,),
                },
            },
            'MLP': {
                'Dense_Hidden_0': {
                    'kernel': (self.d, mlp_hidden),
                    'bias': (mlp_hidden,),
                },
                'Dense_Output': {
                    'kernel': (mlp_hidden, self.d),
                    'bias': (self.d,),
                },
            },
            'ConditionalNorm_MLP': {
                'Dense_0': {
                    'kernel': (self.c, self.d * 2),
                    'bias': (self.d * 2,),
                },
            },
            'attn': {
                'Dense_Q': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_K': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_V': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_Output': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'norm_qk_scale': (1, 1, 1, 1),
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)

  def test_zero_init_is_identity(self):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jax.random.normal(self.key, input_shape)
    cond = jnp.zeros(cond_shape)
    module = dit_blocks.DiTBlockAdaLNZero(hidden_size=self.d, num_heads=4)
    variables = module.init(self.key, x, cond, is_training=False)
    output = module.apply(variables, x, cond, is_training=False)
    self.assertTrue(jnp.allclose(output, x, atol=1e-5))

  def test_output_shape_with_cross_cond(self):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    cross_cond_shape = (self.batch, 8, self.d)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    cross_cond = jnp.ones(cross_cond_shape)
    module = dit_blocks.DiTBlockAdaLNZero(hidden_size=self.d, num_heads=4)
    variables = module.init(
        self.key, x, cond, is_training=False, cross_cond=cross_cond
    )
    output = module.apply(
        variables, x, cond, is_training=False, cross_cond=cross_cond
    )
    self.assertEqual(output.shape, input_shape)

  def test_variable_shapes_with_cross_cond(self):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    cross_cond_shape = (self.batch, 8, self.d)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    cross_cond = jnp.ones(cross_cond_shape)
    mlp_hidden = int(self.d * 4.0)
    module = dit_blocks.DiTBlockAdaLNZero(hidden_size=self.d, num_heads=4)
    variables = module.init(
        self.key, x, cond, is_training=False, cross_cond=cross_cond
    )
    variables_shapes = test_utils.get_pytree_shapes(variables)

    expected_variables_shapes = {
        'params': {
            'Dense_Gate_MSA': {
                'kernel': (self.c, self.d),
                'bias': (self.d,),
            },
            'Dense_Gate_MLP': {
                'kernel': (self.c, self.d),
                'bias': (self.d,),
            },
            'Dense_Gate_Cross': {
                'kernel': (self.c, self.d),
                'bias': (self.d,),
            },
            'ConditionalNorm': {
                'Dense_0': {
                    'kernel': (self.c, self.d * 2),
                    'bias': (self.d * 2,),
                },
            },
            'MLP': {
                'Dense_Hidden_0': {
                    'kernel': (self.d, mlp_hidden),
                    'bias': (mlp_hidden,),
                },
                'Dense_Output': {
                    'kernel': (mlp_hidden, self.d),
                    'bias': (self.d,),
                },
            },
            'attn': {
                'Dense_Q': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_K': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_V': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_Output': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'norm_qk_scale': (1, 1, 1, 1),
            },
            'cross_attn': {
                'Dense_Q': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_K': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_V': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_Output': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'norm_qk_scale': (1, 1, 1, 1),
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)

  def test_zero_init_is_identity_with_cross_cond(self):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    cross_cond_shape = (self.batch, 8, self.d)
    x = jax.random.normal(self.key, input_shape)
    cond = jnp.zeros(cond_shape)
    cross_cond = jax.random.normal(self.key, cross_cond_shape)
    module = dit_blocks.DiTBlockAdaLNZero(hidden_size=self.d, num_heads=4)
    variables = module.init(
        self.key, x, cond, is_training=False, cross_cond=cross_cond
    )
    output = module.apply(
        variables, x, cond, is_training=False, cross_cond=cross_cond
    )
    self.assertTrue(jnp.allclose(output, x, atol=1e-5))


class PositionalEmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch, self.n, self.d = 2, 16, 32

  def test_output_shape(self):
    input_shape = (self.batch, self.n, self.d)
    x = jnp.ones(input_shape)
    module = dit_blocks.PositionalEmbedding()
    variables = module.init(self.key, x)
    output = module.apply(variables, x)
    self.assertEqual(output.shape, input_shape)

  def test_variable_shapes(self):
    input_shape = (self.batch, self.n, self.d)
    x = jnp.ones(input_shape)
    module = dit_blocks.PositionalEmbedding()
    variables = module.init(self.key, x)
    variables_shapes = test_helpers.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'PositionalEmbeddingTensor': (1, self.n, self.d),
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)


class PatchifyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch, self.h, self.w, self.c = 2, 16, 16, 3
    self.patch_size = (4, 4)
    self.embedding_dim = 64

  def test_output_shape(self):
    x = jnp.ones((self.batch, self.h, self.w, self.c))
    module = dit_blocks.Patchify(
        patch_size=self.patch_size, embedding_dim=self.embedding_dim
    )
    variables = module.init(self.key, x)
    output = module.apply(variables, x)
    expected_n = (self.h // self.patch_size[0]) * (self.w // self.patch_size[1])
    self.assertEqual(output.shape, (self.batch, expected_n, self.embedding_dim))

  def test_raises_error_on_non_divisible_shape(self):
    x = jnp.ones((self.batch, self.h + 1, self.w, self.c))
    module = dit_blocks.Patchify(
        patch_size=self.patch_size, embedding_dim=self.embedding_dim
    )
    with self.assertRaises(
        ValueError,
        msg=(
            f'Height {self.h} must be divisible by patch height'
            f' {self.patch_size[0]}.Width {self.w} must be divisible by patch'
            f' width {self.patch_size[1]}.'
        ),
    ):
      module.init(self.key, x)

  def test_variable_shapes(self):
    x = jnp.ones((self.batch, self.h, self.w, self.c))
    module = dit_blocks.Patchify(
        patch_size=self.patch_size, embedding_dim=self.embedding_dim
    )
    variables = module.init(self.key, x)
    variables_shapes = test_helpers.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'Dense_Project': {
                'kernel': (
                    self.patch_size[0] * self.patch_size[1] * self.c,
                    self.embedding_dim,
                ),
                'bias': (self.embedding_dim,),
            }
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)


class DePatchifyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch, self.h, self.w, self.c = 2, 16, 16, 3
    self.patch_size = (4, 4)
    self.embedding_dim = 64
    self.cond_dim = 32

  def test_output_shape(self):
    n = (self.h // self.patch_size[0]) * (self.w // self.patch_size[1])
    x = jnp.ones((self.batch, n, self.embedding_dim))
    cond = jnp.ones((self.batch, self.cond_dim))
    module = dit_blocks.DePatchify(
        patch_size=self.patch_size, output_shape=(self.h, self.w, self.c)
    )
    variables = module.init(self.key, x, cond)
    output = module.apply(variables, x, cond)
    self.assertEqual(output.shape, (self.batch, self.h, self.w, self.c))

  def test_variable_shapes(self):
    n = (self.h // self.patch_size[0]) * (self.w // self.patch_size[1])
    x = jnp.ones((self.batch, n, self.embedding_dim))
    cond = jnp.ones((self.batch, self.cond_dim))
    module = dit_blocks.DePatchify(
        patch_size=self.patch_size, output_shape=(self.h, self.w, self.c)
    )
    variables = module.init(self.key, x, cond)
    variables_shapes = test_helpers.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
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
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)


if __name__ == '__main__':
  absltest.main()
