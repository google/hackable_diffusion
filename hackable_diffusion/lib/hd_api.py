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

"""Foundational Contracts for Hackable Diffusion (Tier 1 Protocols).

This file defines the core interfaces and data structures that constitute the
Hackable Diffusion API. It acts as "Tier 1" layer, establishing the high-level
contracts between different components of the diffusion system.

In scope ("Tier 1" api):

- High-level Interfaces between components for core diffusion concepts
   (e.g., `CorruptionProcess`, `InferenceFn`, `SamplerStep`, `DiffusionLoss`).
- Data Structures used to pass information between components
   (e.g., `StepInfo`, `DiffusionStep`, ...).

Off scope ("Tier 2" api):

- Polymorphic behavior / implementations
    (e.g., specific schedules, specific model architectures, ...).
- Complex logic, internal helper functions, or algorithms.

Focusing this layer on high-level contracts ensures that core pipelines remain
agnostic to specific modeling choices, data types or neural architectures.
"""

from __future__ import annotations

from typing import Protocol

import flax.struct
from hackable_diffusion.lib import hd_typing

###############################################################################
# MARK: Type Aliases
###############################################################################

PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree
Int = hd_typing.Int

DataArray = hd_typing.DataArray
TimeArray = hd_typing.TimeArray

TargetInfo = hd_typing.TargetInfo
ScheduleKey = hd_typing.ScheduleKey
LossOutput = hd_typing.LossOutput

Conditioning = hd_typing.Conditioning


###############################################################################
# MARK: Corruption Process
###############################################################################


class CorruptionSchedule(Protocol):
  """Protocol for corruption schedules."""

  def evaluate(
      self, time: TimeArray  # pyrefly: ignore[not-a-type]
  ) -> dict[ScheduleKey, TimeArray]:  # pyrefly: ignore[not-a-type]
    """Evaluate the schedule for a given time. Return a dictionary of info."""


class CorruptionProcess(Protocol):
  """Base class for all corruption processes (continuous and discrete)."""

  @property
  def schedule(self) -> CorruptionSchedule:
    ...

  def corrupt(
      self,
      key: PRNGKey,
      x0: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> tuple[DataArray, TargetInfo]:  # pyrefly: ignore[not-a-type]
    """Corrupt x0 according to time, and return xt and targets info."""

  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Sample from the invariant distribution."""

  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfo:
    """Convert the prediction to the target type."""


###############################################################################
# MARK: Inference
###############################################################################


class InferenceFn(Protocol):
  """A protocol for an inference function.

  The InferenceFn is responsible for predicting the clean data `x0` from the
  noisy input `xt`. It also predicts related quantities such as ['eps', 'score',
  'velocity', 'v'] in the case of a Gaussian diffusion model. It can also take
  into account guidance and/or conditioning.
  """

  def __call__(
      self, time: TimeArray, xt: DataArray, conditioning: Conditioning | None  # pyrefly: ignore[not-a-type]
  ) -> TargetInfo:  # pyrefly: ignore[not-a-type]
    ...


###############################################################################
# MARK: Training
###############################################################################


class DiffusionLoss(Protocol):
  """Protocol for diffusion loss functions."""

  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> LossOutput:  # pyrefly: ignore[not-a-type]
    """Compute the diffusion loss (no averaging)."""


###############################################################################
# MARK: Sampling
###############################################################################


@flax.struct.dataclass(frozen=True, kw_only=True)
class StepInfo:
  """Holds metadata for the current diffusion step.

  Attributes:
    step: The step number.
    time: The time at which the step is computed.
    rng: The random number generator key.

  All these fields are static and are computed before starting the sampling
    loop.
  """

  step: Int  # pyrefly: ignore[not-a-type]
  time: TimeArray  # pyrefly: ignore[not-a-type]
  rng: PRNGKey


@flax.struct.dataclass(frozen=True, kw_only=True)
class DiffusionStep:
  """The complete state of the diffusion process at a single step.

  Note that in the case where our data structure is a PyTree, the diffusion step
  is defined for each leaf. The associated PyTree is `DiffusionStepTree`.

  Attributes:
    xt: The noisy data at the current step.
    step_info: The `StepInfo` used to compute the current step.
    aux: Additional data computed by the sampler.
  """

  xt: DataArray  # pyrefly: ignore[not-a-type]
  step_info: StepInfo
  aux: PyTree  # pyrefly: ignore[not-a-type]


class SamplerStep(Protocol):
  """A protocol defining the diffusion sampling algorithm (e.g., DDIM)."""

  def initialize(
      self,
      initial_noise: DataArray,  # pyrefly: ignore[not-a-type]
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    """Initializes the step state (e.g. from pure noise)."""
    ...

  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """Performs one step of the sampling process to compute the next state."""
    ...

  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    """Performs the final step to produce the clean output sample."""
    ...


class SampleFn(Protocol):
  """A protocol for a sampling function (high-level orchestration)."""

  def __call__(
      self,
      inference_fn: InferenceFn,
      rng: PRNGKey,
      initial_noise: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
  ) -> tuple[DiffusionStep, DiffusionStep | None]:  # pyrefly: ignore[not-a-type]
    ...
