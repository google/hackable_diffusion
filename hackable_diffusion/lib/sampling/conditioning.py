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

"""Conditioning update protocols and implementations for the sampling loop."""

import dataclasses
from typing import Protocol

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
import jax

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree
Conditioning = hd_typing.Conditioning
DiffusionStep = hd_api.DiffusionStep


def _is_diffusion_leaf(x: PyTree) -> bool:  # pyrefly: ignore[not-a-type]
  """Returns True if the leaf is a DiffusionStep."""
  return isinstance(x, hd_api.DiffusionStep)


################################################################################
# MARK: Protocols
################################################################################


class UpdateConditioningFn(Protocol):
  """Protocol for updating conditioning during the sampling loop.

  This allows injecting step-dependent information back into the conditioning
  dict between sampling steps (e.g. self-conditioning logits from the
  previous prediction).
  """

  def __call__(
      self,
      conditioning: Conditioning,
      step_carry: PyTree[DiffusionStep],  # pyrefly: ignore[not-a-type]
  ) -> Conditioning:
    """Update conditioning based on the current diffusion step.

    Args:
      conditioning: The current conditioning dict.
      step_carry: The current diffusion step state.

    Returns:
      The updated conditioning dict.
    """
    ...


################################################################################
# MARK: Implementations
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class PropagateSelfConditioningFn(UpdateConditioningFn):
  """Propagates predicted logits into the conditioning dict across sampling steps.

  Extracts ``step_carry.aux['logits']`` from the previous diffusion step and
  stores it in ``conditioning[self_conditioning_key]``. Compatible with both
  single-modal (``DiffusionStep``) and multi-modal (``PyTree[DiffusionStep]``)
  samplers.
  """

  self_conditioning_key: str = 'self_conditioning_logits'

  def __call__(
      self,
      conditioning: Conditioning | None,
      step_carry: PyTree[DiffusionStep],  # pyrefly: ignore[not-a-type]
  ) -> Conditioning:
    sc_logits = jax.tree.map(
        lambda step: step.aux['logits'],
        step_carry,
        is_leaf=_is_diffusion_leaf,
    )
    base_cond = dict(conditioning) if conditioning is not None else {}
    base_cond[self.self_conditioning_key] = sc_logits
    return base_cond
