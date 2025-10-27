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

"""Tests for base corruption processes."""

from unittest import mock
import chex
from hackable_diffusion.lib.corruption import base
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized


def _create_leaf_process(data_array, time_array, target_info_name):
  """Creates a process for a pytree leaf."""
  process = mock.MagicMock()
  target_info = {target_info_name: data_array + 5}
  process.corrupt.return_value = (data_array + 5.0, target_info)
  process.sample_from_invariant.return_value = data_array - 1.0
  process.convert_predictions.return_value = target_info
  process.get_schedule_info.return_value = {'time': time_array - 7.0}
  return process


class BaseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('batch_size_1', 1),
      ('batch_size_16', 16),
  )
  def test_nested_process(self, batch_size: int = 16):
    data_tree = {
        'a': {
            'b': jnp.ones((batch_size, 1)),
            'c': {
                'd': jnp.ones((batch_size, 5, 7, 11)),
                'e': jnp.ones((batch_size, 13)),
            },
        },
    }
    time_tree = {
        'a': {
            'b': jnp.ones((batch_size, 1)) * 0.3,
            'c': {
                'd': jnp.ones((batch_size, 5, 7, 11)) * 0.5,
                'e': jnp.ones((batch_size, 13)) * 0.7,
            },
        },
    }
    target_info_tree_names = {
        'a': {
            'b': 'x0',
            'c': {'d': 'score', 'e': 'velocity'},
        },
    }
    processes_tree = jax.tree.map(
        _create_leaf_process, data_tree, time_tree, target_info_tree_names
    )
    key = jax.random.PRNGKey(0)
    nested_process = base.NestedProcess(processes=processes_tree)
    xt, target_info = nested_process.corrupt(key, data_tree, time_tree)
    invariant_out = nested_process.sample_from_invariant(key, data_tree)
    convert_predictions_out = nested_process.convert_predictions(
        target_info, xt, time_tree
    )
    schedule_info = nested_process.get_schedule_info(time_tree)

    expected_invariant_out = {
        'a': {
            'b': jnp.ones((batch_size, 1)) - 1.0,
            'c': {
                'd': jnp.ones((batch_size, 5, 7, 11)) - 1.0,
                'e': jnp.ones((batch_size, 13)) - 1.0,
            },
        },
    }
    expected_convert_predictions_out = {
        'a': {
            'b': {'x0': jnp.ones((batch_size, 1)) + 5},
            'c': {
                'd': {'score': jnp.ones((batch_size, 5, 7, 11)) + 5},
                'e': {'velocity': jnp.ones((batch_size, 13)) + 5},
            },
        },
    }
    expected_schedule_info = {
        'a': {
            'b': {'time': jnp.ones((batch_size, 1)) * 0.3 - 7.0},
            'c': {
                'd': {'time': jnp.ones((batch_size, 5, 7, 11)) * 0.5 - 7.0},
                'e': {'time': jnp.ones((batch_size, 13)) * 0.7 - 7.0},
            },
        },
    }

    chex.assert_trees_all_close(expected_invariant_out, invariant_out)
    chex.assert_trees_all_close(
        expected_convert_predictions_out, convert_predictions_out
    )
    chex.assert_trees_all_close(expected_schedule_info, schedule_info)


if __name__ == '__main__':
  absltest.main()
