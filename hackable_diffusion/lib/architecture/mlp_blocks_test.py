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
    self.assertEqual(output.shape, (self.batch_size, output_size))

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
    self.assertEqual(output.shape, (self.batch_size, self.seq_len, output_size))

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


class SwiGLUTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.is_training = True
    self.input_dim = 48
    self.ff_size = 32
    self.output_size = 48
    self.seq_len = 17
    self.x = jnp.ones((self.batch_size, self.input_dim))
    self.sequence_x = jnp.ones((self.batch_size, self.seq_len, self.input_dim))

  def test_swiglu_output_shape(self):
    """Tests the output shape of the SwiGLU."""
    module = mlp_blocks.SwiGLU(
        hidden_size=self.output_size,
        ff_size=self.ff_size,
    )
    variables = module.init(self.key, self.x, is_training=self.is_training)
    output = module.apply(variables, self.x, is_training=self.is_training)
    self.assertEqual(output.shape, (self.batch_size, self.output_size))

  def test_swiglu_sequence_output_shape(self):
    """Tests the output shape of SwiGLU for sequential input."""
    module = mlp_blocks.SwiGLU(
        hidden_size=self.output_size,
        ff_size=self.ff_size,
    )
    variables = module.init(
        self.key, self.sequence_x, is_training=self.is_training
    )
    output = module.apply(
        variables, self.sequence_x, is_training=self.is_training
    )
    self.assertEqual(
        output.shape, (self.batch_size, self.seq_len, self.output_size)
    )

  def test_swiglu_zero_init_output(self):
    """Tests that zero_init_output produces a zero output."""
    module = mlp_blocks.SwiGLU(
        hidden_size=self.output_size,
        ff_size=self.ff_size,
        zero_init_output=True,
    )
    variables = module.init(self.key, self.x, is_training=self.is_training)
    output = module.apply(variables, self.x, is_training=self.is_training)
    self.assertTrue(jnp.all(output == 0))

  def test_swiglu_sequence_zero_init_output(self):
    """Tests zero_init_output produces a zero output for sequential input."""
    module = mlp_blocks.SwiGLU(
        hidden_size=self.output_size,
        ff_size=self.ff_size,
        zero_init_output=True,
    )
    variables = module.init(
        self.key, self.sequence_x, is_training=self.is_training
    )
    output = module.apply(
        variables, self.sequence_x, is_training=self.is_training
    )
    self.assertTrue(jnp.all(output == 0))

  def test_swiglu_variables_shape(self):
    """Tests SwiGLU variables shape."""
    module = mlp_blocks.SwiGLU(
        hidden_size=self.output_size,
        ff_size=self.ff_size,
    )
    variables = module.init(self.key, self.x, is_training=self.is_training)
    variables_shapes = test_helpers.get_pytree_shapes(variables)
    expected_variables_shapes = {
        'params': {
            'Dense_Up': {
                'kernel': (self.input_dim, self.ff_size * 2),
            },
            'Dense_Down': {
                'kernel': (self.ff_size, self.output_size),
            },
        }
    }
    self.assertDictEqual(expected_variables_shapes, variables_shapes)

  def test_swiglu_no_bias(self):
    """Tests that SwiGLU Dense layers have no bias."""
    module = mlp_blocks.SwiGLU(
        hidden_size=self.output_size,
        ff_size=self.ff_size,
    )
    variables = module.init(self.key, self.x, is_training=self.is_training)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
    for path in leaves_with_paths:
      self.assertNotIn('bias', path)


if __name__ == '__main__':
  absltest.main()
