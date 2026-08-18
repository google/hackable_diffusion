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
from hackable_diffusion.lib.architecture import dit_blocks
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import attention
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


class DiTBlockTest(parameterized.TestCase):
  """Tests for unified DiTBlock module."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch, self.n, self.d, self.c = 2, 16, 32, 64

  @parameterized.named_parameters(
      dict(
          testcase_name='rms_norm_swiglu',
          norm_strategy=normalization.ConditionalRMSNormStrategy(
              use_shift=False
          ),
          use_gates=False,
          ffn_type='swiglu',
      ),
      dict(
          testcase_name='rms_norm_dense',
          norm_strategy=normalization.ConditionalRMSNormStrategy(
              use_shift=False
          ),
          use_gates=False,
          ffn_type='dense',
      ),
      dict(
          testcase_name='ln_zero_swiglu',
          norm_strategy=normalization.ConditionalLayerNormStrategy(
              use_shift=True,
          ),
          use_gates=True,
          ffn_type='swiglu',
      ),
      dict(
          testcase_name='ln_zero_dense',
          norm_strategy=normalization.ConditionalLayerNormStrategy(
              use_shift=True,
          ),
          use_gates=True,
          ffn_type='dense',
      ),
  )
  def test_output_shape(self, norm_strategy, use_gates, ffn_type):
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    module = dit_blocks.DiTBlock(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
        norm_strategy=norm_strategy,
        use_gates=use_gates,
        ffn_type=ffn_type,
    )
    variables = module.init(self.key, x, cond, is_training=False)
    output = module.apply(variables, x, cond, is_training=False)
    self.assertEqual(output.shape, input_shape)  # pyrefly: ignore[missing-attribute]

  @parameterized.named_parameters(
      dict(
          testcase_name='rms_norm',
          norm_strategy=normalization.ConditionalRMSNormStrategy(
              use_shift=False
          ),
          use_gates=False,
      ),
      dict(
          testcase_name='ln_zero',
          norm_strategy=normalization.ConditionalLayerNormStrategy(
              use_shift=True,
          ),
          use_gates=True,
      ),
  )
  def test_zero_init_is_identity(self, norm_strategy, use_gates):
    """Tests identity-at-init."""
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jax.random.normal(self.key, input_shape)
    cond = jnp.zeros(cond_shape)
    module = dit_blocks.DiTBlock(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
        norm_strategy=norm_strategy,
        use_gates=use_gates,
    )
    variables = module.init(self.key, x, cond, is_training=False)
    output = module.apply(variables, x, cond, is_training=False)
    self.assertTrue(jnp.allclose(output, x, atol=1e-5))  # pyrefly: ignore[bad-argument-type]

  def test_variable_shapes_ada_rms_norm(self):
    """Tests variable shapes with ada_rms_norm (SwiGLU)."""
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    mlp_hidden = int(self.d * 4.0)
    module = dit_blocks.DiTBlock(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
        norm_strategy=normalization.ConditionalRMSNormStrategy(use_shift=False),
        use_gates=False,
        ffn_type='swiglu',
    )
    variables = module.init(self.key, x, cond, is_training=False)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]

    expected_variables_shapes = {
        'params': {
            'ConditionalNorm_Attention': {
                'conditioning': {
                    'Dense_Scale': {
                        'kernel': (self.c, self.d),
                        'bias': (self.d,),
                    },
                },
            },
            'ConditionalNorm_MLP': {
                'conditioning': {
                    'Dense_Scale': {
                        'kernel': (self.c, self.d),
                        'bias': (self.d,),
                    },
                },
            },
            'ffn': {
                'Dense_Up': {
                    'kernel': (self.d, mlp_hidden * 2),
                },
                'Dense_Down': {
                    'kernel': (mlp_hidden, self.d),
                },
            },
            'attention': {
                'Dense_Q': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_K': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_V': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_Output': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'norm_qk_scale': (1, 1, 1, 1),
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)

  def test_variable_shapes_ada_ln_zero(self):
    """Tests variable shapes with ada_ln_zero (GELU)."""
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    mlp_hidden = int(self.d * 4.0)
    module = dit_blocks.DiTBlock(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
        norm_strategy=normalization.ConditionalLayerNormStrategy(
            use_shift=True,
        ),
        use_gates=True,
        ffn_type='dense',
    )
    variables = module.init(self.key, x, cond, is_training=False)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]

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
                'conditioning': {
                    'Dense_ScaleShift': {
                        'kernel': (self.c, self.d * 2),
                        'bias': (self.d * 2,),
                    },
                },
            },
            'ffn': {
                'Dense_Up': {
                    'kernel': (self.d, mlp_hidden),
                },
                'Dense_Down': {
                    'kernel': (mlp_hidden, self.d),
                },
            },
            'ConditionalNorm_MLP': {
                'conditioning': {
                    'Dense_ScaleShift': {
                        'kernel': (self.c, self.d * 2),
                        'bias': (self.d * 2,),
                    },
                },
            },
            'attention': {
                'Dense_Q': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_K': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_V': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'Dense_Output': {'kernel': (self.d, self.d), 'bias': (self.d,)},
                'norm_qk_scale': (1, 1, 1, 1),
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)

  # MARK: Validation error tests

  def test_use_gates_false_without_zero_init_output_raises(self):
    """Verifies that use_gates=False with zero_init_output=False raises."""
    module = dit_blocks.DiTBlock(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
        norm_strategy=normalization.ConditionalRMSNormStrategy(use_shift=False),
        use_gates=False,
        zero_init_output=False,
    )
    x = jnp.ones((self.batch, self.n, self.d))
    cond = jnp.ones((self.batch, self.c))
    with self.assertRaisesRegex(
        ValueError, 'zero_init_output must be True when use_gates is False'
    ):
      module.init(self.key, x, cond, is_training=False)

  # MARK: ffn_use_bias tests

  @parameterized.named_parameters(
      ('swiglu_no_bias', 'swiglu', False),
      ('swiglu_with_bias', 'swiglu', True),
      ('dense_no_bias', 'dense', False),
      ('dense_with_bias', 'dense', True),
  )
  def test_ffn_use_bias(self, ffn_type, ffn_use_bias):
    """Verifies that ffn_use_bias controls bias in the FFN sub-module."""
    x = jnp.ones((self.batch, self.n, self.d))
    cond = jnp.ones((self.batch, self.c))
    module = dit_blocks.DiTBlock(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
        norm_strategy=normalization.ConditionalRMSNormStrategy(use_shift=False),
        use_gates=False,
        ffn_type=ffn_type,
        ffn_use_bias=ffn_use_bias,
    )
    variables = module.init(self.key, x, cond, is_training=False)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)

    ffn_bias_paths = [
        p
        for p in leaves_with_paths
        if p.startswith('params/ffn/') and 'bias' in p
    ]
    if ffn_use_bias:
      self.assertLen(ffn_bias_paths, 2)  # Dense_Up/bias + Dense_Down/bias
    else:
      self.assertEmpty(ffn_bias_paths)


class DiTBlockPresetsTest(parameterized.TestCase):
  """Tests for DiTBlock preset subclasses."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch, self.n, self.d, self.c = 2, 16, 32, 64

  @parameterized.named_parameters(
      ('flux', dit_blocks.DiTBlockFlux),
      ('sd3', dit_blocks.DiTBlockSD3),
      ('ada_ln_zero', dit_blocks.DiTBlockAdaLNZero),
  )
  def test_preset_output_shape(self, block_cls):
    """Tests that preset subclasses produce the correct output shape."""
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jnp.ones(input_shape)
    cond = jnp.ones(cond_shape)
    module = block_cls(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
    )
    variables = module.init(self.key, x, cond, is_training=False)
    output = module.apply(variables, x, cond, is_training=False)
    self.assertEqual(output.shape, input_shape)

  @parameterized.named_parameters(
      ('flux', dit_blocks.DiTBlockFlux),
      ('sd3', dit_blocks.DiTBlockSD3),
      ('ada_ln_zero', dit_blocks.DiTBlockAdaLNZero),
  )
  def test_preset_identity_at_init(self, block_cls):
    """Tests that preset subclasses satisfy identity-at-init."""
    input_shape = (self.batch, self.n, self.d)
    cond_shape = (self.batch, self.c)
    x = jax.random.normal(self.key, input_shape)
    cond = jnp.zeros(cond_shape)
    module = block_cls(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
    )
    variables = module.init(self.key, x, cond, is_training=False)
    output = module.apply(variables, x, cond, is_training=False)
    self.assertTrue(jnp.allclose(output, x, atol=1e-5))

  def test_flux_has_no_gates(self):
    """Verifies DiTBlockFlux has no gate parameters."""
    x = jnp.ones((self.batch, self.n, self.d))
    cond = jnp.ones((self.batch, self.c))
    module = dit_blocks.DiTBlockFlux(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
    )
    variables = module.init(self.key, x, cond, is_training=False)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
    gate_paths = [p for p in leaves_with_paths if 'Gate' in p]
    self.assertEmpty(gate_paths)

  def test_sd3_has_gates(self):
    """Verifies DiTBlockSD3 has gate parameters."""
    x = jnp.ones((self.batch, self.n, self.d))
    cond = jnp.ones((self.batch, self.c))
    module = dit_blocks.DiTBlockSD3(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
    )
    variables = module.init(self.key, x, cond, is_training=False)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
    gate_paths = [p for p in leaves_with_paths if 'Gate' in p]
    self.assertNotEmpty(gate_paths)

  def test_ada_ln_zero_has_gates(self):
    """Verifies DiTBlockAdaLNZero has gate parameters."""
    x = jnp.ones((self.batch, self.n, self.d))
    cond = jnp.ones((self.batch, self.c))
    module = dit_blocks.DiTBlockAdaLNZero(
        hidden_size=self.d,
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=4),
    )
    variables = module.init(self.key, x, cond, is_training=False)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
    gate_paths = [p for p in leaves_with_paths if 'Gate' in p]
    self.assertNotEmpty(gate_paths)


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
    self.assertEqual(output.shape, input_shape)  # pyrefly: ignore[missing-attribute]

  def test_variable_shapes(self):
    input_shape = (self.batch, self.n, self.d)
    x = jnp.ones(input_shape)
    module = dit_blocks.PositionalEmbedding()
    variables = module.init(self.key, x)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]
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
    self.assertEqual(output.shape, (self.batch, expected_n, self.embedding_dim))  # pyrefly: ignore[missing-attribute]

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
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]
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

  def test_output_shape(self):
    n = (self.h // self.patch_size[0]) * (self.w // self.patch_size[1])
    x = jnp.ones((self.batch, n, self.embedding_dim))
    module = dit_blocks.DePatchify(
        patch_size=self.patch_size, output_shape=(self.h, self.w, self.c)
    )
    variables = module.init(self.key, x)
    output = module.apply(variables, x)
    self.assertEqual(output.shape, (self.batch, self.h, self.w, self.c))  # pyrefly: ignore[missing-attribute]

  def test_variable_shapes(self):
    n = (self.h // self.patch_size[0]) * (self.w // self.patch_size[1])
    x = jnp.ones((self.batch, n, self.embedding_dim))
    module = dit_blocks.DePatchify(
        patch_size=self.patch_size,
        output_shape=(self.h, self.w, self.c),
    )
    variables = module.init(self.key, x)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]
    expected_variables_shapes = {
        'params': {
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
