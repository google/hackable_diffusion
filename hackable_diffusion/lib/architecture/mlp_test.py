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

"""Tests for the MLP module."""

from hackable_diffusion.lib import test_utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import mlp
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Type Aliases
################################################################################

ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Tests
################################################################################


class MLPTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.is_training = True
    self.shape = (4, 4, 3)
    self.prod_shape = int(np.prod(self.shape))
    self.cond_dim = 16
    self.x = jnp.ones((self.batch_size, *self.shape))
    self.flatten_x = jnp.reshape(self.x, (self.batch_size, -1))
    self.concatenate_emb = {
        ConditioningMechanism.CONCATENATE: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    self.sum_emb = {
        ConditioningMechanism.SUM: jnp.ones((self.batch_size, self.cond_dim)),
    }

  # ConditionalMLP tests

  def test_conditional_mlp_output_shape(self):
    """Tests the output shape of the MLP."""
    mlp_module = mlp.ConditionalMLP(
        hidden_sizes_preprocess=[32, 16],
        hidden_sizes_postprocess=[32, 16],
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=False,
        conditioning_mechanism=ConditioningMechanism.CONCATENATE,
    )
    variables = mlp_module.init(
        self.key,
        self.x,
        self.concatenate_emb,
        is_training=self.is_training,
    )
    output = mlp_module.apply(
        variables,
        self.x,
        self.concatenate_emb,
        is_training=self.is_training,
    )
    self.assertEqual(output.shape, self.x.shape)

  def test_conditional_mlp_zero_init_output(self):
    """Tests that zero_init_output produces a zero output."""
    mlp_module = mlp.ConditionalMLP(
        hidden_sizes_preprocess=[32, 16],
        hidden_sizes_postprocess=[32, 16],
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=True,
        conditioning_mechanism=ConditioningMechanism.CONCATENATE,
    )
    variables = mlp_module.init(
        self.key,
        self.x,
        self.concatenate_emb,
        is_training=self.is_training,
    )
    output = mlp_module.apply(
        variables,
        self.x,
        self.concatenate_emb,
        is_training=self.is_training,
    )
    self.assertTrue(jnp.all(output == 0))

  def test_conditional_mlp_concatenate_variables_shape(self):
    """Tests MLP variables shape."""
    input_dim = int(np.prod(self.shape))
    hidden_sizes_preprocess = [32, 16]
    hidden_sizes_postprocess = [32, 16]

    hidden_sizes_preprocess_x_dim = [input_dim] + hidden_sizes_preprocess

    concatenate_dim = hidden_sizes_preprocess[-1] * 2

    hidden_sizes_postprocess_dim = [concatenate_dim] + hidden_sizes_postprocess

    mlp_module = mlp.ConditionalMLP(
        hidden_sizes_preprocess=[32, 16],
        hidden_sizes_postprocess=[32, 16],
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=False,
        conditioning_mechanism=ConditioningMechanism.CONCATENATE,
    )
    variables = mlp_module.init(
        self.key,
        self.x,
        self.concatenate_emb,
        is_training=self.is_training,
    )

    leaves_with_paths = test_utils.get_leaves_with_paths(variables)
    expected_shapes = dict()
    for i in range(len(hidden_sizes_preprocess) - 1):
      name_prefix = f'params/PreprocessMLP/Dense_Hidden_{i}'
      expected_shapes[f'{name_prefix}/kernel'] = (
          hidden_sizes_preprocess_x_dim[i],
          hidden_sizes_preprocess_x_dim[i + 1],
      )
      expected_shapes[f'{name_prefix}/bias'] = (
          hidden_sizes_preprocess_x_dim[i + 1],
      )
    name_prefix = 'params/PreprocessMLP/Dense_Output'
    expected_shapes[f'{name_prefix}/kernel'] = (
        hidden_sizes_preprocess_x_dim[-2],
        hidden_sizes_preprocess_x_dim[-1],
    )
    expected_shapes[f'{name_prefix}/bias'] = (
        hidden_sizes_preprocess_x_dim[-1],
    )
    for i in range(len(hidden_sizes_postprocess)):
      name_prefix = f'params/PostprocessMLP/Dense_Hidden_{i}'
      expected_shapes[f'{name_prefix}/kernel'] = (
          hidden_sizes_postprocess_dim[i],
          hidden_sizes_postprocess_dim[i + 1],
      )
      expected_shapes[f'{name_prefix}/bias'] = (
          hidden_sizes_postprocess_dim[i + 1],
      )
    name_prefix = 'params/PostprocessMLP/Dense_Output'
    expected_shapes[f'{name_prefix}/kernel'] = (
        hidden_sizes_postprocess_dim[-1],
        input_dim,
    )
    expected_shapes[f'{name_prefix}/bias'] = (input_dim,)

    for path, leaf in leaves_with_paths.items():
      self.assertIn(path, expected_shapes)
      self.assertEqual(leaf.shape, expected_shapes[path])

  def test_conditional_mlp_sum_variables_shape(self):
    """Tests MLP variables shape."""
    input_dim = int(np.prod(self.shape))
    hidden_sizes_preprocess = [32, 16]
    hidden_sizes_postprocess = [32, 16]

    hidden_sizes_preprocess_x_dim = [input_dim] + hidden_sizes_preprocess

    sum_dim = hidden_sizes_preprocess[-1]

    hidden_sizes_postprocess_dim = [sum_dim] + hidden_sizes_postprocess

    mlp_module = mlp.ConditionalMLP(
        hidden_sizes_preprocess=[32, 16],
        hidden_sizes_postprocess=[32, 16],
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=False,
        conditioning_mechanism=ConditioningMechanism.SUM,
    )
    variables = mlp_module.init(
        self.key,
        self.x,
        self.sum_emb,
        is_training=self.is_training,
    )

    leaves_with_paths = test_utils.get_leaves_with_paths(variables)
    expected_shapes = dict()
    for i in range(len(hidden_sizes_preprocess) - 1):
      name_prefix = f'params/PreprocessMLP/Dense_Hidden_{i}'
      expected_shapes[f'{name_prefix}/kernel'] = (
          hidden_sizes_preprocess_x_dim[i],
          hidden_sizes_preprocess_x_dim[i + 1],
      )
      expected_shapes[f'{name_prefix}/bias'] = (
          hidden_sizes_preprocess_x_dim[i + 1],
      )
    name_prefix = 'params/PreprocessMLP/Dense_Output'
    expected_shapes[f'{name_prefix}/kernel'] = (
        hidden_sizes_preprocess_x_dim[-2],
        hidden_sizes_preprocess_x_dim[-1],
    )
    expected_shapes[f'{name_prefix}/bias'] = (
        hidden_sizes_preprocess_x_dim[-1],
    )

    name_prefix = 'params/Dense_Projection_Conditioning'
    expected_shapes[f'{name_prefix}/kernel'] = (
        self.cond_dim,
        hidden_sizes_preprocess_x_dim[-1],
    )
    expected_shapes[f'{name_prefix}/bias'] = (
        hidden_sizes_preprocess_x_dim[-1],
    )

    for i in range(len(hidden_sizes_postprocess)):
      name_prefix = f'params/PostprocessMLP/Dense_Hidden_{i}'
      expected_shapes[f'{name_prefix}/kernel'] = (
          hidden_sizes_postprocess_dim[i],
          hidden_sizes_postprocess_dim[i + 1],
      )
      expected_shapes[f'{name_prefix}/bias'] = (
          hidden_sizes_postprocess_dim[i + 1],
      )
    name_prefix = 'params/PostprocessMLP/Dense_Output'
    expected_shapes[f'{name_prefix}/kernel'] = (
        hidden_sizes_postprocess_dim[-1],
        input_dim,
    )
    expected_shapes[f'{name_prefix}/bias'] = (input_dim,)

    for path, leaf in leaves_with_paths.items():
      self.assertIn(path, expected_shapes)
      self.assertEqual(leaf.shape, expected_shapes[path])


if __name__ == '__main__':
  absltest.main()
