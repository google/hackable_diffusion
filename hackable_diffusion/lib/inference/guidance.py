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
from typing import Protocol, TypeAlias
from hackable_diffusion.lib import corruption
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import jax_helpers
import jax
import jax.numpy as jnp


################################################################################
# MARK: Type Aliases
################################################################################

Batch: TypeAlias = hd_api.Batch
Conditioning: TypeAlias = hd_api.Conditioning
typechecked = jax_helpers.typechecked

################################################################################
# MARK: Protocols
################################################################################


class GuidanceFn[Operand](Protocol):
  """Guidance function protocol."""

  def __call__(
      self,
      xt: hd_api.Data[Operand],
      conditioning: Conditioning | None,
      time: hd_api.Time[Operand],
      cond_outputs: hd_api.TargetInfo[Operand],
      uncond_outputs: hd_api.TargetInfo[Operand],
  ) -> hd_api.TargetInfo[Operand]:
    """Combine conditional and unconditional outputs."""
    ...


################################################################################
# MARK: Guidance Functions
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class ScalarGuidanceFn(GuidanceFn[Batch]):
  """Simple scalar guidance function."""

  guidance: float = 0.0

  @typechecked
  def __call__(
      self,
      xt: hd_api.Data[Batch],
      conditioning: Conditioning | None,
      time: hd_api.Time[Batch],
      cond_outputs: hd_api.TargetInfo[Batch],
      uncond_outputs: hd_api.TargetInfo[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    """Simple scalar guidance function."""
    del conditioning, time, xt  # unused
    return jax.tree.map(
        lambda x, y: x * (1.0 + self.guidance) - y * self.guidance,
        cond_outputs,
        uncond_outputs,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class LimitedIntervalGuidanceFn(GuidanceFn[Batch]):
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

  def _interval_mask(self, time: hd_api.Time[Batch]) -> jax.Array:
    """Returns a boolean mask indicating whether time/metric is within bounds."""
    return jnp.logical_and(time >= self.lower, time <= self.upper)

  @typechecked
  def __call__(
      self,
      xt: hd_api.Data[Batch],
      conditioning: Conditioning | None,
      time: hd_api.Time[Batch],
      cond_outputs: hd_api.TargetInfo[Batch],
      uncond_outputs: hd_api.TargetInfo[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    """Limited interval guidance function."""
    del conditioning, xt  # unused

    local_guidance = jnp.where(
        self._interval_mask(time),
        self.guidance,
        0.0,
    )

    return jax.tree.map(
        lambda x, y: x * (1.0 + local_guidance) - y * local_guidance,
        cond_outputs,
        uncond_outputs,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class LogSnrLimitedIntervalGuidanceFn(LimitedIntervalGuidanceFn):
  """Limited interval guidance function based on logsnr."""

  noise_schedule: corruption.GaussianSchedule

  def _interval_mask(self, time: hd_api.Time[Batch]) -> jax.Array:
    """Returns a boolean mask indicating whether time/metric is within bounds."""
    logsnr = self.noise_schedule.logsnr(time)
    return jnp.logical_and(logsnr >= self.lower, logsnr <= self.upper)
