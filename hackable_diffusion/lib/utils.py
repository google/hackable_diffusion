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

from typing import Any, Callable

import jax
import jax.numpy as jnp
import jaxtyping

import numpy as np


# Type Aliases
Key = jaxtyping.Key


def tree_map_with_key(
    fn: Callable[..., Any],
    key: Key,
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
