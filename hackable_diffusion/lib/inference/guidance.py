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

"""Guidance."""

import dataclasses
from typing import Protocol
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp


################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree

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
      xt: DataTree,
      conditioning: Conditioning,
      time: TimeTree,
      cond_outputs: TargetInfoTree,
      uncond_outputs: TargetInfoTree,
  ) -> TargetInfoTree:
    """Combine conditional and unconditional outputs."""
    ...


################################################################################
# MARK: Guidance Functions
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class ScalarGuidanceFn(GuidanceFn):
  """Simple scalar guidance function."""

  guidance: float = 0.0

  @typechecked
  def __call__(
      self,
      xt: DataTree,
      conditioning: Conditioning,
      time: TimeTree,
      cond_outputs: TargetInfoTree,
      uncond_outputs: TargetInfoTree,
  ) -> TargetInfoTree:
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

  @typechecked
  def __call__(
      self,
      xt: DataArray,
      conditioning: Conditioning,
      time: TimeArray,
      cond_outputs: TargetInfo,
      uncond_outputs: TargetInfo,
  ) -> TargetInfo:
    """Simple scalar guidance function."""
    del conditioning  # unused
    time = utils.bcast_right(time, xt.ndim)

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


################################################################################
# MARK: Nested Guidance
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedGuidanceFn(GuidanceFn):
  """Nested guidance function."""

  guidance_fns: PyTree[GuidanceFn]

  @typechecked
  def __call__(
      self,
      xt: DataTree,
      conditioning: Conditioning,
      time: TimeTree,
      cond_outputs: TargetInfoTree,
      uncond_outputs: TargetInfoTree,
  ) -> TargetInfoTree:
    """Combine conditional and unconditional outputs."""
    return jax.tree.map(
        lambda guidance_fn, xt, time, cond_out, uncond_out: guidance_fn(
            xt=xt,
            conditioning=conditioning,
            time=time,
            cond_outputs=cond_out,
            uncond_outputs=uncond_out,
        ),
        self.guidance_fns,
        xt,
        time,
        cond_outputs,
        uncond_outputs,
    )
