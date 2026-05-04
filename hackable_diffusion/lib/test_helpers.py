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

"""Various testing utilities."""

from typing import Any, Sequence

import flax
from flax import linen as nn

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
import jax


################################################################################
# MARK: Type Aliases
################################################################################

VariableDict = flax.core.scope.VariableDict
DictKey = jax.tree_util.DictKey
Float = hd_typing.Float
GetAttrKey = jax.tree_util.GetAttrKey
Path = Sequence[DictKey | GetAttrKey]

################################################################################
# MARK: Path utilities
################################################################################


def _get_path(sequence_dict_keys: Path) -> str:
  """Returns the path of a leaf node in a dictionary.

  Args:
    sequence_dict_keys: The path to a leaf node in a dictionary.

  Returns:
    The path of the leaf node in a dictionary.
  """
  sequence_dict_keys = list(sequence_dict_keys)
  keys = [
      element.key for element in sequence_dict_keys if hasattr(element, 'key')
  ]
  if len(keys) < len(sequence_dict_keys) - 1:
    raise ValueError(
        'At most only one key can be a GetAttrKey (and therefore dropped from'
        ' the path).'
    )
  return '/'.join(keys)


def get_leaves_with_paths(variables: VariableDict) -> dict[str, Any]:
  """Returns a dictionary of leaves with paths in a dictionary.

  Usage:
    variables = {'params': {'kernel': 1.0, 'bias': 2.0}}
    leaves_with_paths = get_leaves_with_paths(variables)
    self.assertEqual(leaves_with_paths, {'params/kernel': 1.0, 'params/bias':
    2.0})

  Args:
    variables: The dictionary of variables.

  Returns:
    A dictionary of leaves with paths in a dictionary.
  """

  leaves_with_paths = jax.tree.leaves_with_path(variables)
  return {_get_path(path): array for path, array in leaves_with_paths}


def get_pytree_shapes(pytree: dict[str, Any]) -> dict[str, Any]:
  """Recursively extracts the shape of every leaf in a PyTree."""
  return jax.tree_util.tree_map(lambda x: getattr(x, 'shape', None), pytree)


class IdentityBackbone(nn.Module, arch_typing.ConditionalBackbone):

  @nn.compact
  def __call__(
      self,
      x: arch_typing.DataTree,
      conditioning_embeddings: arch_typing.ConditioningEmbeddings,
      is_training: bool,
  ) -> arch_typing.DataTree:
    return x
