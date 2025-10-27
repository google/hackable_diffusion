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

"""Tests for the MLP blocks."""

from hackable_diffusion.lib import test_utils
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
    self.x = jnp.ones((self.batch_size, *self.shape))
    self.flatten_x = jnp.reshape(self.x, (self.batch_size, -1))

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
    leaves_with_paths = test_utils.get_leaves_with_paths(variables)
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


if __name__ == '__main__':
  absltest.main()
