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

"""Utility functions."""

import functools
from types import FunctionType  # pylint: disable=g-importing-member
from typing import Any, Callable, cast
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp
import numpy as np

################################################################################
# MARK: Type Aliases
################################################################################

Array = hd_typing.Array
PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

PointwiseFn = Callable[[Array["..."]], Array["..."]]
PointwiseMethod = Callable[[Any, Array["..."]], Array["..."]]

DataTree = hd_typing.DataTree
ShapeTree = hd_typing.ShapeTree
DType = hd_typing.DType
DTypeTree = hd_typing.DTypeTree

################################################################################
# MARK: Utils
################################################################################


def lenient_map(
    fn: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
):
  """Like jax.tree.map but with a lenient structure matching.

  The PyTree structure of the output is determined by the structure of `tree`.
  The structures of `rest` are used only to determine the leaf values to be
  mapped.

  Example usage:
    a = [1.0, 2.0]
    b = (5.0, 6.0)
    c = lenient_map(lambda x, y: x+y, a, b)
    # c is [6.0, 8.0]

  If one were to use jax.tree.map directly, one would get an error because the
  structure of `a` is not the same as the structure of `b`.

  Args:
    fn: The function to apply to each leaf.
    tree: The tree to map.
    *rest: Additional arguments to pass to fn.
    is_leaf: A function that takes a leaf and returns True if it should be
      mapped. If None, all leaves are mapped.

  Returns:
    The tree resulting from applying fn to each leaf in `tree`.

  Raises:
    KeyError: If the structures of `tree` and `rest` do not match.
  """
  path_vals, struct = jax.tree.flatten_with_path(tree, is_leaf=is_leaf)
  paths, _ = zip(*path_vals)
  restructured_rest = []
  for r in rest:
    r_path_vals, r_struct = jax.tree.flatten_with_path(r, is_leaf=is_leaf)
    r_paths, r_leaves = zip(*r_path_vals)

    if r_paths != paths:
      raise KeyError(f"Paths of the trees must match. But {paths} != {r_paths}")

    restructured_rest.append(jax.tree.unflatten(struct, r_leaves))
    del r_struct
  return jax.tree.map(fn, tree, *restructured_rest)


@typechecked
def tree_map_with_key(
    fn: Callable[..., Any],
    key: PRNGKey,
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
  """Like jax.tree.map but with a separate PRNG key for each leaf.

  Arguments:
    fn: The function to apply to each leaf. Takes the key as the first arg, i.e.
      of the form `fn(key: PRNGKey, tree_leaf: Any, *rest_leafs: Any) -> Any`.
    key: The PRNG key from which to split all the leaf-keys.
    tree: The tree to map.
    *rest: Additional arguments to pass to fn.
    is_leaf: A function that takes a leaf and returns True if it should be
      mapped. If None, all leaves are mapped.

  Returns:
    The tree resulting from applying fn to each leaf.
  """

  def _with_folded_key(path, *args) -> Any:
    # Hash the path to get a unique key per array.
    # The hash of a path is a 64-bit integer, but fold_in expects a
    # non-negative 32-bit integer. We use a bitwise AND with 0xffffffff to
    # truncate the hash to 32 bits, which results in a Python integer within
    # the valid uint32 range.
    key_x = jax.random.fold_in(key, jnp.uint32(hash(path) & 0xFFFFFFFF))
    return fn(key_x, *args)

  return jax.tree.map_with_path(_with_folded_key, tree, *rest, is_leaf=is_leaf)


def get_broadcastable_shape(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
  """Return a shape of ones except for the given axes which keep the shape.

  Example:
    _broadcastable_shape((2, 5, 7), (0,)) -> (2, 1, 1)
    _broadcastable_shape((2, 5, 7), (0, 2)) -> (2, 1, 7)

  Args:
    shape: The shape to be broadcastable to.
    axes: The axes to keep the shape of.

  Returns:
    A shape that is broadcastable to the input shape with ones everywhere except
    for axes.
  """
  axes = np.lib.array_utils.normalize_axis_tuple(axes, len(shape))
  return tuple(s if i in axes else 1 for i, s in enumerate(shape))


def egrad(g):
  """Returns a function that returns the elementwise gradient of a function.

  More details: https://github.com/jax-ml/jax/issues/3556#issuecomment-649779759

  Args:
    g: The function to take the element-wise gradient of.

  Returns:
    A wrapper around the function g that returns the element-wise gradient.
  """

  def wrapped(x):
    y, g_vjp = jax.vjp(g, x)
    (x_bar,) = g_vjp(jnp.ones_like(y))
    return x_bar

  return wrapped


@typechecked
def flatten_non_batch_dims(x: Array["batch ..."]):
  """Reshapes the array with `B` as the first dimension to (B, ...).

  If the array has only one batched dimension, the result has shape (B, 1). If
  the array has the shape (B, W, H, C), the result has shape (B, W * H * C).

  Args:
    x: The array to flatten.

  Returns:
    The flattened array.
  """
  return x.reshape((x.shape[0], -1))


@typechecked
def bcast_right(value: Array["*shape"], ndim: int) -> Array["*out_shape"]:
  """Broadcast by adding singleton axes to the right, instead of to the left."""
  if value.ndim > ndim:
    raise ValueError(
        f"Cannot reverse broadcast a value with {value.ndim} dimensions "
        f"to {ndim} dimensions."
    )

  if value.ndim < ndim:
    difference = ndim - value.ndim
    return value.reshape(value.shape + difference * (1,))
  else:
    return value


_to_bf16_from_fp32 = (
    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x
)
_to_fp32_from_bf16 = (
    lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x
)


def to_bf16_from_fp32(x: PyTree) -> PyTree:
  """Converts a PyTree of arrays to bfloat16.

  Leaves of the PyTree are converted to bfloat16 if they are float32.
  If the leaf is not float32, it is unchanged.

  Args:
    x: The PyTree of arrays to convert.

  Returns:
    The PyTree of arrays converted to bfloat16.
  """
  return jax.tree.map(_to_bf16_from_fp32, x)


def optional_bf16_to_fp32(x: PyTree) -> PyTree:
  """Converts a PyTree of arrays from bfloat16.

  Leaves of the PyTree are converted to float32 if they are bfloat16.
  If the leaf is not bfloat16, it is unchanged.

  Args:
    x: The PyTree of arrays to convert.

  Returns:
    The PyTree of arrays converted from bfloat16.
  """
  return jax.tree.map(_to_fp32_from_bf16, x)


class CustomGradient:
  """A convenience decorator to define a custom gradient for a function or method.

  Simplified interface compared to `jax.custom_jvp` but with the following
  limitations:
    - assumes a single input argument
    - assumes that the derivative takes the same input argument

  Usage:
  ```
  # for a standalone function:
  @CustomGradient
  def my_function(x):
      return jnp.square(x)

  @my_function.derivative
  def my_function_der(x):
      return 2 * x


  # for a class method:
  @dataclasses.dataclass
  class MyClass:
    a : int = 3

    @CustomGradient
    def my_method(self, x):
        return x ** self.a  # can use self here

    @my_method.derivative
    def my_method_der(self, x):
        return self.a * x ** (self.a - 1)
  ```

  It works on functions and methods, by being both a callable (for functions)
  and a descriptor (for methods), lazily creating the appropriate JAX VJP rule
  on first use.
  """

  def __init__(self, primal_fn: PointwiseFn | PointwiseMethod):
    # cast to function type to stop pytype from complaining about
    # self.primal_fn.__name__
    self.primal_fn: FunctionType = cast(FunctionType, primal_fn)
    self.derivative_fn = None

  def derivative(
      self, derivative_fn: PointwiseFn | PointwiseMethod
  ) -> PointwiseFn | PointwiseMethod:
    """Decorator to store the derivative function."""
    if self.derivative_fn:
      raise ValueError(
          f"Derivative already defined for {self.primal_fn.__name__}"
      )
    self.derivative_fn: FunctionType = cast(FunctionType, derivative_fn)
    return derivative_fn  # return decorated function

  @functools.cached_property
  def _vjp_for_function(self):
    """Builds the jax.custom_vjp implementation for a standalone function."""
    primal_wrapper = jax.custom_vjp(self.primal_fn)

    def f_fwd(arg):
      return self.primal_fn(arg), arg

    def f_bwd(arg, g):
      grad_val = g * self.derivative_fn(arg)
      return (grad_val,)

    primal_wrapper.defvjp(f_fwd, f_bwd)
    return primal_wrapper

  @functools.cached_property
  def _vjp_for_method(self):
    """Builds the jax.custom_vjp implementation for a class method."""
    # The VJP is defined on the original (unbound) method
    primal_wrapper = jax.custom_vjp(self.primal_fn, nondiff_argnames=("self",))

    def f_fwd(self_, arg):
      return self.primal_fn(self_, arg), arg

    def f_bwd(self_, arg, g):
      grad_val = g * self.derivative_fn(self_, arg)
      return (grad_val,)

    primal_wrapper.defvjp(f_fwd, f_bwd)
    return primal_wrapper

  def __call__(self, arg):
    """Handles the case of a decorated standalone function."""
    if self.derivative_fn is None:
      raise ValueError(f"Derivative not defined for {self.primal_fn.__name__}")

    return self._vjp_for_function(arg)

  def __get__(self, instance, objtype=None):
    """Handles the case where the decorator is used on a method."""
    if instance is None:
      return self  # Accessed on the class not on an instance
    if self.derivative_fn is None:
      raise ValueError(f"Derivative not defined for {self.primal_fn.__name__}")
    # Bind the instance ('self') to the VJP-wrapped method
    return functools.partial(self._vjp_for_method, instance)


################################################################################
# MARK: Utils for batch
################################################################################


def is_tuple_of_ints(obj):
  return isinstance(obj, tuple) and all(isinstance(item, int) for item in obj)


def get_dummy_batch_fixed_dtype(
    shape: ShapeTree,
    dtype: DType,
    only_first_axis: bool = False,
) -> DataTree:
  """Get a dummy batch of data with a fixed dtype.

  `shape` is a PyTree of shapes. For each leaf of `shape`, we use the fixed
  dtype is used to determine the dtype of the dummy array.

  Args:
    shape: The shape of the dummy batch.
    dtype: The dtype of the dummy batch.
    only_first_axis: If True, we only use the first axis of the shape to
      determine the dimension of the dummy array.

  Returns:
    A dummy batch of data.
  """

  def _get_dummy(shape: tuple[int, ...]) -> jnp.ndarray:
    if only_first_axis:
      return jnp.empty(shape=(shape[0],), dtype=dtype)
    else:
      return jnp.empty(shape=shape, dtype=dtype)

  return jax.tree.map(
      _get_dummy,
      shape,
      is_leaf=is_tuple_of_ints,
  )


def get_dummy_batch(
    shape: ShapeTree,
    dtype: DTypeTree,
    only_first_axis: bool = False,
) -> DataTree:
  """Get a dummy batch of data.

  `shape` and `dtype` are PyTrees with the same structure. For each leaf of
  `shape`, the corresponding dtype in `dtype` is used to determine the dtype of
  the dummy array.

  Args:
    shape: The shape of the dummy batch.
    dtype: The dtype of the dummy batch.
    only_first_axis: If True, we only use the first axis of the shape to
      determine the dimension of the dummy array.

  Returns:
    A dummy batch of data.
  """

  def _get_dummy(shape: tuple[int, ...], dtype: DType) -> jnp.ndarray:
    if only_first_axis:
      return jnp.empty(shape=(shape[0],), dtype=dtype)
    else:
      return jnp.empty(shape=shape, dtype=dtype)

  return jax.tree.map(
      _get_dummy,
      shape,
      dtype,
      is_leaf=is_tuple_of_ints,
  )


################################################################################
# MARK: Utils for random numbers
################################################################################


@typechecked
def jax_randint(
    key: PRNGKey,
    minval: int = 0,
    maxval: int = jnp.iinfo(jnp.int32).max,
) -> int:
  return jax.random.randint(key, shape=(), minval=minval, maxval=maxval).item()
