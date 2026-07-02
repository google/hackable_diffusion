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

"""Tests for the MLP blocks."""

from hackable_diffusion.lib import test_helpers
from hackable_diffusion.lib.architecture import mlp_blocks
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class MLPBlocksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.is_training = True
    self.shape = (4, 4, 3)
    self.seq_len = 17
    self.x = jnp.ones((self.batch_size, *self.shape))
    self.sequence_x = jnp.ones((self.batch_size, self.seq_len, *self.shape))
    self.flatten_x = jnp.reshape(self.x, (self.batch_size, -1))
    self.flatten_sequence_x = jnp.reshape(
        self.sequence_x, (self.batch_size, self.seq_len, -1)
    )

  # MLP tests

  def test_mlp_output_shape(self):
    """Tests the output shape of the MLP."""
    output_size = 3
    mlp_module = mlp_blocks.MLP(
        hidden_sizes=[32, 16],
        output_size=output_size,
        activation='relu',
        dropout_rate=0.0,
    )
    variables = mlp_module.init(
        self.key,
        self.flatten_x,
        is_training=self.is_training,
    )
    output = mlp_module.apply(
        variables,
        self.flatten_x,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, (self.batch_size, output_size))  # pyrefly: ignore[missing-attribute]

  def test_mlp_sequence_output_shape(self):
    """Tests the output shape of the MLP for sequential input."""

    output_size = 3
    mlp_module = mlp_blocks.MLP(
        hidden_sizes=[32, 16],
        output_size=output_size,
        activation='relu',
        dropout_rate=0.0,
    )
    variables = mlp_module.init(
        self.key,
        self.flatten_sequence_x,
        is_training=self.is_training,
    )
    output = mlp_module.apply(
        variables,
        self.flatten_sequence_x,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, (self.batch_size, self.seq_len, output_size))  # pyrefly: ignore[missing-attribute]

  def test_mlp_zero_init_output(self):
    """Tests that zero_init_output produces a zero output."""
    mlp_module = mlp_blocks.MLP(
        hidden_sizes=[32, 16],
        output_size=3,
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=True,
    )
    variables = mlp_module.init(
        self.key,
        self.flatten_x,
        is_training=self.is_training,
    )
    output = mlp_module.apply(
        variables,
        self.flatten_x,
        is_training=self.is_training,
    )
    self.assertTrue(jnp.all(output == 0))

  def test_mlp_sequence_zero_init_output(self):
    """Tests zero_init_output produces a zero output for sequential input."""
    mlp_module = mlp_blocks.MLP(
        hidden_sizes=[32, 16],
        output_size=3,
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=True,
    )
    variables = mlp_module.init(
        self.key,
        self.flatten_sequence_x,
        is_training=self.is_training,
    )
    output = mlp_module.apply(
        variables,
        self.flatten_sequence_x,
        is_training=self.is_training,
    )
    self.assertTrue(jnp.all(output == 0))

  def test_mlp_variables_shape(self):
    """Tests MLP variables shape."""
    input_dim = int(np.prod(self.shape))
    hidden_layers = [32, 16]
    all_layers = [input_dim] + hidden_layers
    output_size = 3
    mlp_module = mlp_blocks.MLP(
        hidden_sizes=[32, 16],
        output_size=3,
        activation='relu',
        dropout_rate=0.0,
    )
    variables = mlp_module.init(
        self.key,
        self.flatten_x,
        is_training=self.is_training,
    )
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
    expected_shapes = dict()
    for i in range(len(hidden_layers)):
      name_prefix = f'params/Dense_Hidden_{i}'
      expected_shapes[f'{name_prefix}/kernel'] = (
          all_layers[i],
          all_layers[i + 1],
      )
      expected_shapes[f'{name_prefix}/bias'] = (all_layers[i + 1],)
    name_prefix = 'params/Dense_Output'
    expected_shapes[f'{name_prefix}/kernel'] = (
        all_layers[-1],
        output_size,
    )
    expected_shapes[f'{name_prefix}/bias'] = (output_size,)
    for path, leaf in leaves_with_paths.items():
      self.assertIn(path, expected_shapes)
      self.assertEqual(leaf.shape, expected_shapes[path])

  def test_mlp_sequence_variables_shape(self):
    """Tests MLP variables shape for sequential input."""
    input_dim = int(np.prod(self.shape))
    hidden_layers = [32, 16]
    all_layers = [input_dim] + hidden_layers
    output_size = 3
    mlp_module = mlp_blocks.MLP(
        hidden_sizes=[32, 16],
        output_size=3,
        activation='relu',
        dropout_rate=0.0,
    )
    variables = mlp_module.init(
        self.key,
        self.flatten_sequence_x,
        is_training=self.is_training,
    )
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
    expected_shapes = dict()
    for i in range(len(hidden_layers)):
      name_prefix = f'params/Dense_Hidden_{i}'
      expected_shapes[f'{name_prefix}/kernel'] = (
          all_layers[i],
          all_layers[i + 1],
      )
      expected_shapes[f'{name_prefix}/bias'] = (all_layers[i + 1],)
    name_prefix = 'params/Dense_Output'
    expected_shapes[f'{name_prefix}/kernel'] = (
        all_layers[-1],
        output_size,
    )
    expected_shapes[f'{name_prefix}/bias'] = (output_size,)
    for path, leaf in leaves_with_paths.items():
      self.assertIn(path, expected_shapes)
      self.assertEqual(leaf.shape, expected_shapes[path])


class GatingSwiGLUTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.input_dim = 48
    self.features = 32
    self.x = jnp.ones((self.batch_size, self.input_dim))

  def test_linear_swiglu_output_shape(self):
    """Tests the output shape of GatingSwiGLU sub-layer."""
    module = mlp_blocks.GatingSwiGLU(features=self.features)
    variables = module.init(self.key, self.x)
    output = module.apply(variables, self.x)
    self.assertEqual(output.shape, (self.batch_size, self.features))  # pyrefly: ignore[missing-attribute]

  def test_linear_swiglu_variables_shape(self):
    """Tests GatingSwiGLU parameter tree tracking configurations."""
    module = mlp_blocks.GatingSwiGLU(features=self.features, use_bias=False)
    variables = module.init(self.key, self.x)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]
    expected_variables_shapes = {
        'params': {
            'Dense_Gate_Val': {
                'kernel': (self.input_dim, self.features * 2),
            }
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)


class FeedForwardTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.output_size = 48
    self.hidden_size = 64
    self.seq_len = 17
    self.x = jnp.ones((self.batch_size, self.output_size))
    self.sequence_x = jnp.ones(
        (self.batch_size, self.seq_len, self.output_size)
    )

  @parameterized.named_parameters(
      ('swiglu_mode', 'swiglu', 'gelu'),  # activation is ignored
      ('dense_mode_gelu', 'dense', 'gelu'),
      ('dense_mode_silu', 'dense', 'silu'),
  )
  def test_feedforward_output_shapes(
      self, ffn_type: mlp_blocks.FFNType, activation: str
  ):
    """Tests the flat and sequential output shapes across architectures."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        ffn_type=ffn_type,
        activation=activation,
    )

    # Flat Input
    variables = module.init(self.key, self.x, is_training=True)
    output = module.apply(variables, self.x, is_training=True)
    self.assertEqual(output.shape, (self.batch_size, self.output_size))  # pyrefly: ignore[missing-attribute]

    # Sequential Input
    output_seq = module.apply(variables, self.sequence_x, is_training=True)
    self.assertEqual(
        output_seq.shape, (self.batch_size, self.seq_len, self.output_size)  # pyrefly: ignore[missing-attribute]
    )

  @parameterized.named_parameters(
      ('swiglu_mode', 'swiglu'),
      ('dense_mode', 'dense'),
  )
  def test_feedforward_zero_init_output(self, ffn_type: mlp_blocks.FFNType):
    """Verifies that terminal weight projections respect zero initialization."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        ffn_type=ffn_type,
        zero_init_output=True,
    )
    variables = module.init(self.key, self.x, is_training=True)
    output = module.apply(variables, self.x, is_training=True)
    self.assertTrue(jnp.all(output == 0))

  def test_feedforward_swiglu_parameter_structure(self):
    """Ensures SwiGLU pipelines decouple biases on projection maps."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        ffn_type='swiglu',
    )
    variables = module.init(self.key, self.x, is_training=True)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)

    # Assert true SwiGLU implementation contains no biases anywhere
    for path in leaves_with_paths:
      self.assertNotIn('bias', path)

    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]

    # Updated to match the flattened inline footprint precisely
    expected_shapes = {
        'params': {
            'Dense_Up': {
                'kernel': (self.output_size, self.hidden_size * 2),
            },
            'Dense_Down': {
                'kernel': (self.hidden_size, self.output_size),
            },
        }
    }
    self.assertDictEqual(expected_shapes, variables_shapes)

  def test_feedforward_dense_parameter_structure(self):
    """Ensures dense layout has no biases by default (use_bias=False)."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        ffn_type='dense',
    )
    variables = module.init(self.key, self.x, is_training=True)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]
    expected_shapes = {
        'params': {
            'Dense_Up': {
                'kernel': (self.output_size, self.hidden_size),
            },
            'Dense_Down': {
                'kernel': (self.hidden_size, self.output_size),
            },
        }
    }
    self.assertDictEqual(expected_shapes, variables_shapes)

  @parameterized.named_parameters(
      ('swiglu_with_bias', 'swiglu'),
      ('dense_with_bias', 'dense'),
  )
  def test_feedforward_use_bias_true(self, ffn_type: mlp_blocks.FFNType):
    """Ensures use_bias=True adds bias to both up and down projections."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        ffn_type=ffn_type,
        use_bias=True,
    )
    variables = module.init(self.key, self.x, is_training=True)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)

    # Both Dense_Up and Dense_Down should have bias
    bias_paths = [p for p in leaves_with_paths if 'bias' in p]
    self.assertLen(bias_paths, 2)

  def test_feedforward_dropout_handling(self):
    """Verifies internal regularization scales states deterministically."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        dropout_rate=0.5,
    )
    rng1, rng2, rng_drop = jax.random.split(self.key, 3)
    x_rand = jax.random.normal(rng1, (self.batch_size, self.output_size))

    variables = module.init(rng2, x_rand, is_training=False)

    # Eval mode should be identical
    out_eval_1 = module.apply(variables, x_rand, is_training=False)
    out_eval_2 = module.apply(variables, x_rand, is_training=False)
    np.testing.assert_allclose(out_eval_1, out_eval_2, atol=1e-6)  # pyrefly: ignore[no-matching-overload]

    # Train mode activation scaling (with .item() wrapping)
    out_train = module.apply(
        variables, x_rand, is_training=True, rngs={'dropout': rng_drop}
    )

    max_train_val = jnp.max(jnp.abs(out_train))  # pyrefly: ignore[bad-argument-type]
    max_eval_val = jnp.max(jnp.abs(out_eval_1))  # pyrefly: ignore[bad-argument-type]

    self.assertGreater(
        float(max_train_val.item()),
        float(max_eval_val.item()),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='swiglu_float32', ffn_type='swiglu', dtype=jnp.float32
      ),
      dict(
          testcase_name='swiglu_bfloat16', ffn_type='swiglu', dtype=jnp.bfloat16
      ),
      dict(
          testcase_name='dense_float32',
          ffn_type='dense',
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='dense_bfloat16',
          ffn_type='dense',
          dtype=jnp.bfloat16,
      ),
  )
  def test_feedforward_output_dtype(
      self, ffn_type: mlp_blocks.FFNType, dtype: jnp.dtype
  ):
    """Verifies the output dtype matches the configured dtype."""
    module = mlp_blocks.FeedForward(
        output_size=self.output_size,
        hidden_size=self.hidden_size,
        ffn_type=ffn_type,
        dtype=dtype,
    )
    x = jnp.ones((self.batch_size, self.output_size), dtype=dtype)
    variables = module.init(self.key, x, is_training=False)
    output = module.apply(variables, x, is_training=False)
    self.assertEqual(output.dtype, dtype)  # pyrefly: ignore[missing-attribute]


if __name__ == '__main__':
  absltest.main()
