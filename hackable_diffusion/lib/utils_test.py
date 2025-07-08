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

"""Tests for tree_utils."""

from hackable_diffusion.lib import utils
import jax
import jax.numpy as jnp
from absl.testing import absltest


class UtilsTest(absltest.TestCase):

  def test_tree_map_with_key(self):
    key = jax.random.PRNGKey(0)
    tree = {
        "a": jnp.array([1, 2]),
        "b": {"c": jnp.array([3, 4]), "d": jnp.array([5, 6])},
    }
    received_keys = []

    def _record_key(key, arr):
      # convert key to hashable tuple
      received_keys.append(tuple(key.tolist()))
      return arr

    mapped_tree = utils.tree_map_with_key(_record_key, key, tree)

    # check that all keys are unique
    self.assertEqual(len(received_keys), len(set(received_keys)))
    # check that the returned tree is unchanged (_record_key is a no-op)
    self.assertDictEqual(tree, mapped_tree)

  def test_get_broadcastable_shape(self):
    # Test with empty shape and empty batch_axes
    self.assertEqual(utils.get_broadcastable_shape((), ()), ())

    # Test with a shape with one dimension
    self.assertEqual(utils.get_broadcastable_shape((5,), (0,)), (5,))

    # Test negative indexing
    self.assertEqual(utils.get_broadcastable_shape((2, 3, 5), (-1,)), (1, 1, 5))

    # Test with a shape and empty batch_axes
    self.assertEqual(utils.get_broadcastable_shape((2, 3, 4), ()), (1, 1, 1))

    # Test with a shape and one batch_axis
    self.assertEqual(utils.get_broadcastable_shape((2, 3, 4), (0,)), (2, 1, 1))
    self.assertEqual(utils.get_broadcastable_shape((2, 3, 4), (1,)), (1, 3, 1))
    self.assertEqual(utils.get_broadcastable_shape((2, 3, 4), (2,)), (1, 1, 4))

    # Test with a shape and multiple batch_axes
    self.assertEqual(
        utils.get_broadcastable_shape((2, 3, 4), (0, 1)), (2, 3, 1)
    )
    self.assertEqual(
        utils.get_broadcastable_shape((2, 3, 4), (0, 2)), (2, 1, 4)
    )
    self.assertEqual(
        utils.get_broadcastable_shape((2, 3, 4), (1, 2)), (1, 3, 4)
    )
    self.assertEqual(
        utils.get_broadcastable_shape((2, 3, 4), (0, 1, 2)), (2, 3, 4)
    )

  def test_get_broadcastable_shape_raises(self):
    # out of bounds axis
    with self.assertRaisesRegex(IndexError, "out of bounds"):
      utils.get_broadcastable_shape((2, 3, 4), axes=(124,))

    # empty array
    with self.assertRaisesRegex(IndexError, "out of bounds"):
      utils.get_broadcastable_shape((), axes=(0,))

    # duplicate axes
    with self.assertRaisesRegex(ValueError, "repeated axis"):
      utils.get_broadcastable_shape((2, 3, 4), axes=(0, 0))

    # Raises error on effectively duplicate axes
    # (2 and -1 are the same for ndims=3)
    with self.assertRaisesRegex(ValueError, "repeated axis"):
      utils.get_broadcastable_shape((2, 3, 4), axes=(2, -1))


if __name__ == "__main__":
  absltest.main()
