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

"""Commonly used types and protocols for the architecture.

Please refer to individual modules for more detailed documentation and
definitions of the components.
"""

from typing import Any, Callable, Protocol
from hackable_diffusion.lib import hd_typing
import jax


################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float

DataTree = hd_typing.DataTree

################################################################################
# MARK: Constants
################################################################################

INVALID_INT = -1

################################################################################
# MARK: Enums
################################################################################


################################################################################
# MARK: Conditioning Mechanism
################################################################################


# Conditioning embeddings corresponds to a dictionary with keys corresponding to
# the specification of a conditioning mechanism.
# Example of common conditioning mechanisms:
#    - adaptive_norm
#    - cross_attention
#    - concatenate
#    - sum
#    - self_conditioning

ConditioningEmbeddings = dict[str, Any]

################################################################################
# MARK: Types and protocols
################################################################################

ActivationFn = Callable[[jax.Array], jax.Array]


class ConditionalBackbone(Protocol):
  """Protocol for a conditional backbone."""

  def __call__(
      self,
      x: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning_embeddings: ConditioningEmbeddings,
      *,
      is_training: bool,
  ) -> DataTree:  # pyrefly: ignore[not-a-type]
    ...
