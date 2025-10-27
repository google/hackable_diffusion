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

"""Tests for utils."""

import abc
import dataclasses
import itertools
from typing import Any, Protocol

import chex
from flax import linen as nn
from flax.core import frozen_dict
from hackable_diffusion.lib import utils
import jax
import jax.numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-multiple-import,g-importing-member

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Tests
################################################################################


class UtilsTest(parameterized.TestCase):

  ##############################################################################
  # MARK: Test for flatten_non_batch_dims
  ##############################################################################

  @parameterized.named_parameters(
      ('complex_array', (2, 3, 4), (2, 3 * 4)),
      ('simple_array', (2,), (2, 1)),
  )
  def test_flatten_non_batch_dims_correct_on_complex_arrays(
      self, input_shape, expected_shape
  ):
    a = jnp.ones(input_shape)
    flattened = utils.flatten_non_batch_dims(a)
    self.assertEqual(expected_shape, flattened.shape)

  ##############################################################################
  # MARK: Test for tree_map_with_key
  ##############################################################################

  def test_tree_map_with_key(self):
    key = jax.random.PRNGKey(0)
    tree = {
        'a': jnp.array([1, 2]),
        'b': {'c': jnp.array([3, 4]), 'd': jnp.array([5, 6])},
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
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      utils.get_broadcastable_shape((2, 3, 4), axes=(124,))

    # empty array
    with self.assertRaisesRegex(IndexError, 'out of bounds'):
      utils.get_broadcastable_shape((), axes=(0,))

    # duplicate axes
    with self.assertRaisesRegex(ValueError, 'repeated axis'):
      utils.get_broadcastable_shape((2, 3, 4), axes=(0, 0))

    # Raises error on effectively duplicate axes
    # (2 and -1 are the same for ndims=3)
    with self.assertRaisesRegex(ValueError, 'repeated axis'):
      utils.get_broadcastable_shape((2, 3, 4), axes=(2, -1))

  ##############################################################################
  # MARK: Test for CustomGradient
  ##############################################################################

  def test_manual_gradient_definition(self):

    @utils.CustomGradient
    def value_fn(x):
      return jnp.square(x)

    @value_fn.derivative
    def derivative_fn(x):  # pylint: disable=unused-variable
      return 3.0 * x

    self.assertEqual(value_fn(1.0), 1.0)
    self.assertEqual(jax.grad(value_fn)(1.0), 3.0)
    self.assertEqual(jax.grad(value_fn.primal_fn)(1.0), 2.0)

  def test_egrad(self):
    f = lambda x: x**3
    egrad_f = utils.egrad(f)
    grad_f = jax.grad(f)

    # scalar gradient behave the same.
    self.assertTrue(jnp.allclose(egrad_f(2.0), grad_f(2.0)))

    # element-wise gradient is possible.
    self.assertTrue(
        jnp.allclose(
            egrad_f(jnp.array([1.0, 2.0, 3.0])), jnp.array([3.0, 12.0, 27.0])
        )
    )

    # regular grad fails.
    with self.assertRaisesRegex(
        TypeError, 'Gradient only defined for scalar-output functions'
    ):
      _ = grad_f(jnp.array([1.0, 2.0, 3.0]))

  ##############################################################################
  # MARK: Test for bcast_right
  ##############################################################################

  @parameterized.named_parameters(
      ('broadcast_1d_to_3d', (2,), 3, (2, 1, 1)),
      ('broadcast_2d_to_4d', (2, 3), 4, (2, 3, 1, 1)),
      ('no_broadcast', (2, 3), 2, (2, 3)),
  )
  def test_bcast_right_output_shape(self, input_shape, ndim, expected_shape):
    """Tests bcast_right output shape."""
    value = jnp.ones(input_shape)
    output = utils.bcast_right(value, ndim)
    self.assertEqual(output.shape, expected_shape)

  @parameterized.named_parameters(
      ('not_enough_dims', (2, 3, 4), 2),
      ('negative_ndim', (2, 3, 4), -1),
      ('zero_ndim', (2, 3, 4), 0),
  )
  def test_bcast_right_raises_error(self, shape, ndim):
    """Tests that bcast_right raises an error for invalid ndim."""
    with self.assertRaises(ValueError):
      utils.bcast_right(jnp.ones(shape), ndim)

  ##############################################################################
  # MARK: Test for conversion functions
  ##############################################################################

  def test_to_bf16_from_fp32(self):
    """Tests to_bf16_from_fp32 conversion."""
    tree = {
        'a': jnp.ones((2, 2), dtype=jnp.float32),
        'b': jnp.zeros((3, 3), dtype=jnp.int32),
        'c': (jnp.ones((1,), jnp.float32), jnp.ones((1,), jnp.float16)),
    }
    converted_tree = utils.to_bf16_from_fp32(tree)
    self.assertEqual(converted_tree['a'].dtype, jnp.bfloat16)
    self.assertEqual(converted_tree['b'].dtype, jnp.int32)
    self.assertEqual(converted_tree['c'][0].dtype, jnp.bfloat16)
    self.assertEqual(converted_tree['c'][1].dtype, jnp.float16)

  def test_to_fp32_from_bf16(self):
    """Tests to_fp32_from_bf16 conversion."""
    tree = {
        'a': jnp.ones((2, 2), dtype=jnp.bfloat16),
        'b': jnp.zeros((3, 3), dtype=jnp.int32),
        'c': (jnp.ones((1,), jnp.bfloat16), jnp.ones((1,), jnp.float16)),
    }
    converted_tree = utils.optional_bf16_to_fp32(tree)
    self.assertEqual(converted_tree['a'].dtype, jnp.float32)
    self.assertEqual(converted_tree['b'].dtype, jnp.int32)
    self.assertEqual(converted_tree['c'][0].dtype, jnp.float32)
    self.assertEqual(converted_tree['c'][1].dtype, jnp.float16)

  def test_to_bf16_from_fp32_raises_on_non_array(self):
    """Tests that to_bf16_from_fp32 raises an error for non-array inputs."""
    tree = {'a': jnp.ones((2, 2), dtype=jnp.float32), 'b': 'not_an_array'}
    with self.assertRaises(AttributeError):
      utils.to_bf16_from_fp32(tree)

  ##############################################################################
  # MARK: Test for lenient_map
  ##############################################################################

  @parameterized.named_parameters(
      ('dict', 'dict'),
      ('tuple', 'tuple'),
      ('list', 'list'),
      ('frozen_dict', 'frozen_dict'),
  )
  def test_nested_base(self, input_type: str):

    if input_type == 'dict':
      x = {'a': 1.0, 'b': 2.0}
      a = {'a': 3.0, 'b': 5.0}
      b = {'a': 6.0, 'b': 9.0}
      expected_output = {'a': 10.0, 'b': 16.0}
    elif input_type == 'tuple':
      x = (1.0, 2.0)
      a = (3.0, 5.0)
      b = (6.0, 9.0)
      expected_output = (10.0, 16.0)
    elif input_type == 'list':
      x = [1.0, 2.0]
      a = [3.0, 5.0]
      b = [6.0, 9.0]
      expected_output = [10.0, 16.0]
    elif input_type == 'frozen_dict':
      x = frozen_dict.FrozenDict({'a': 1.0, 'b': 2.0})
      a = frozen_dict.FrozenDict({'a': 3.0, 'b': 5.0})
      b = frozen_dict.FrozenDict({'a': 6.0, 'b': 9.0})
      expected_output = frozen_dict.FrozenDict({'a': 10.0, 'b': 16.0})
    else:
      raise ValueError(f'Unsupported input type: {input_type}')

    class Foo(nn.Module):
      a: PyTree[float]
      b: PyTree[float]

      @nn.compact
      def __call__(self, x):
        return utils.lenient_map(
            lambda x_leaf, a_leaf, b_leaf: x_leaf + a_leaf + b_leaf,
            x,
            self.a,
            self.b,
        )

    foo = Foo(a=a, b=b)
    variables = foo.init(rngs={}, x=x)
    output = foo.apply(variables, x=x)

    chex.assert_trees_all_close(expected_output, output)

  def test_nested_base_for_nn_module(self):

    class BaseEmbedder(nn.Module, abc.ABC):

      @abc.abstractmethod
      def __call__(self, x):
        ...

    class DenseEmbedder(BaseEmbedder):
      num_features: int

      @nn.compact
      def __call__(self, x):
        return nn.Dense(features=self.num_features)(x)

    class NestedEmbedder(BaseEmbedder):
      embedders: PyTree[BaseEmbedder]

      @nn.compact
      def __call__(self, x):
        return utils.lenient_map(lambda x, module: module(x), x, self.embedders)

    embedders = {'a': DenseEmbedder(32), 'b': DenseEmbedder(64)}
    embedder = NestedEmbedder(embedders=embedders)
    x = {'a': jnp.ones((128, 5)), 'b': jnp.ones((128, 7))}
    rng = jax.random.PRNGKey(0)
    variables = embedder.init(rngs={'params': rng}, x=x)
    output = embedder.apply(variables, x=x)

    chex.assert_trees_all_equal_structs(output, x)

  @parameterized.named_parameters(
      ('dict', 'dict'),
      ('tuple', 'tuple'),
      ('list', 'list'),
      ('frozen_dict', 'frozen_dict'),
  )
  def test_no_need_of_nested_wrapper_for_dataclasses(self, input_type: str):

    if input_type == 'dict':
      x = {'a': 1.0, 'b': 2.0}
      a = {'a': 3.0, 'b': 5.0}
      b = {'a': 6.0, 'b': 9.0}
      expected_output = {'a': 10.0, 'b': 16.0}
    elif input_type == 'tuple':
      x = (1.0, 2.0)
      a = (3.0, 5.0)
      b = (6.0, 9.0)
      expected_output = (10.0, 16.0)
    elif input_type == 'list':
      x = [1.0, 2.0]
      a = [3.0, 5.0]
      b = [6.0, 9.0]
      expected_output = [10.0, 16.0]
    elif input_type == 'frozen_dict':
      x = frozen_dict.FrozenDict({'a': 1.0, 'b': 2.0})
      a = frozen_dict.FrozenDict({'a': 3.0, 'b': 5.0})
      b = frozen_dict.FrozenDict({'a': 6.0, 'b': 9.0})
      expected_output = frozen_dict.FrozenDict({'a': 10.0, 'b': 16.0})
    else:
      raise ValueError(f'Unsupported input type: {input_type}')

    class BaseFn(Protocol):

      def __call__(self, x: Any):
        ...

    @dataclasses.dataclass(kw_only=True, frozen=True)
    class NestedFn(BaseFn):
      a: PyTree[float]
      b: PyTree[float]

      def __call__(self, x):
        return jax.tree.map(
            lambda x_leaf, a_leaf, b_leaf: x_leaf + a_leaf + b_leaf,
            x,
            self.a,
            self.b,
        )

    nested_fn = NestedFn(a=a, b=b)
    output = nested_fn(x=x)
    chex.assert_trees_all_close(output, expected_output)

  @parameterized.parameters(
      itertools.product(
          # PyTree
          ['dict', 'list', 'tuple'],
          # fixed_dtype
          [False, True],
          # only_first_axis
          [False, True],
      )
  )
  def test_get_dummy_batch(
      self, input_type: str, fixed_dtype: bool, only_first_axis: bool
  ):

    dtype_fixed = jnp.float32
    if input_type == 'dict':
      if not only_first_axis:
        shape = {'a': (2, 3), 'b': (4, 5)}
      else:
        shape = {'a': (2,), 'b': (4,)}

      dtype = {'a': jnp.float32, 'b': jnp.int32}
      expected_output = {
          'a': jnp.zeros(shape['a'], dtype=dtype['a']),
          'b': jnp.zeros(shape['b'], dtype=dtype['b']),
      }
      expected_output_fixed = {
          'a': jnp.zeros(shape['a'], dtype=dtype_fixed),
          'b': jnp.zeros(shape['b'], dtype=dtype_fixed),
      }
    elif input_type == 'tuple':
      if not only_first_axis:
        shape = ((2, 3), (4, 5))
      else:
        shape = ((2,), (4,))
      dtype = (jnp.float32, jnp.int32)
      expected_output = (
          jnp.zeros(shape[0], dtype=dtype[0]),
          jnp.zeros(shape[1], dtype=dtype[1]),
      )
      expected_output_fixed = (
          jnp.zeros(shape[0], dtype=dtype_fixed),
          jnp.zeros(shape[1], dtype=dtype_fixed),
      )
    elif input_type == 'list':
      if not only_first_axis:
        shape = [(2, 3), (4, 5)]
      else:
        shape = [(2,), (4,)]
      dtype = [jnp.float32, jnp.int32]
      expected_output = [
          jnp.zeros(shape[0], dtype=dtype[0]),
          jnp.zeros(shape[1], dtype=dtype[1]),
      ]
      expected_output_fixed = [
          jnp.zeros(shape[0], dtype=dtype_fixed),
          jnp.zeros(shape[1], dtype=dtype_fixed),
      ]
    else:
      raise ValueError(f'Unsupported input type: {input_type}')

    if fixed_dtype:
      output = utils.get_dummy_batch_fixed_dtype(
          shape, dtype=dtype_fixed, only_first_axis=only_first_axis
      )
      expected_output = expected_output_fixed
    else:
      output = utils.get_dummy_batch(
          shape, dtype=dtype, only_first_axis=only_first_axis
      )
    chex.assert_trees_all_close(output, expected_output)

  def test_lenient_map_fails_on_mismatched_paths(self):
    tree = {'a': 1.0, 'b': 2.0}
    other = {'a': 3.0, 'B': 4.0}  # Mismatched paths.
    with self.assertRaisesRegex(KeyError, 'Paths of the trees must match.'):
      utils.lenient_map(lambda x, y: x + y, tree, other)


if __name__ == '__main__':
  absltest.main()
