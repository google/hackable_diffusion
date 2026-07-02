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

"""Guidance."""

import dataclasses
from typing import Protocol
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
TargetInfo = hd_typing.TargetInfo
TargetInfoTree = hd_typing.TargetInfoTree
TimeArray = hd_typing.TimeArray
TimeTree = hd_typing.TimeTree

################################################################################
# MARK: Protocols
################################################################################


class GuidanceFn(Protocol):
  """Guidance function protocol."""

  def __call__(
      self,
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      cond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      uncond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Combine conditional and unconditional outputs."""
    ...


################################################################################
# MARK: Guidance Functions
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class ScalarGuidanceFn(GuidanceFn):
  """Simple scalar guidance function."""

  guidance: float = 0.0

  @kt.typechecked
  def __call__(
      self,
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      cond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      uncond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Simple scalar guidance function."""
    del conditioning, time, xt  # unused
    return jax.tree.map(
        lambda x, y: x * (1.0 + self.guidance) - y * self.guidance,
        cond_outputs,
        uncond_outputs,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class LimitedIntervalGuidanceFn(GuidanceFn):
  """Limited interval guidance function.

  This follows the guidance function from https://arxiv.org/abs/2404.07724.
  """

  guidance: float = 0.0
  lower: float = 0.0
  upper: float = 1.0

  def __post_init__(self):
    if self.lower >= self.upper:
      raise ValueError(
          "Lower bound must be strictly smaller than the upper bound."
      )

  @kt.typechecked
  def __call__(
      self,
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      cond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      uncond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Limited interval guidance function."""
    del conditioning  # unused
    time = jax_helpers.bcast_right(time, xt.ndim)

    is_in_interval = jnp.logical_and(time >= self.lower, time <= self.upper)
    local_guidance = jnp.where(
        is_in_interval,
        self.guidance,
        0.0,
    )

    return jax.tree.map(
        lambda x, y: x * (1.0 + local_guidance) - y * local_guidance,
        cond_outputs,
        uncond_outputs,
    )
