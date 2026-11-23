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
Hackable Diffusion API. It acts as the "Tier 1" layer, establishing the
high-level contracts between different components of the diffusion system.

In scope ("Tier 1" API):

- High-level interfaces between components for core diffusion concepts
  (e.g., `CorruptionProcess`, `InferenceFn`, `SamplerStep`, `DiffusionLoss`).
- Data structures used to pass information between components
  (e.g., `StepInfo`, `DiffusionStep`, `ScheduleInfo`, `TargetInfo`).

Out of scope ("Tier 2" API):

- Polymorphic behavior / concrete implementations
  (e.g., specific schedules, specific model architectures, ...).
- Complex logic, internal helper functions, or algorithms.

Focusing this layer on high-level contracts ensures that core pipelines remain
agnostic to specific modeling choices, data types, or neural architectures.
"""

from collections.abc import Mapping
from typing import Any, Protocol, TYPE_CHECKING, TypeAlias

import flax.struct
from hackable_diffusion.lib import jax_helpers
import jax
import jaxtyping as jt
import typing_extensions

###############################################################################
# MARK: Operand & Domain Type Aliases
###############################################################################

PRNGKey: TypeAlias = jax_helpers.PRNGKey
Conditioning: TypeAlias = Mapping[str, Any]

if TYPE_CHECKING:
  type Data[Operand] = Operand
  type Time[Operand] = Operand
  type Loss[Operand] = Operand
  Batch: TypeAlias = jax.Array
else:
  # pylint: disable=invalid-name

  @jax_helpers.subscriptable
  def Data(operand: Any) -> Any:
    return operand(jt.Num, '*data')

  @jax_helpers.subscriptable
  def Time(operand: Any) -> Any:
    return operand(jt.Float, '*#data')

  @jax_helpers.subscriptable
  def Loss(operand: Any) -> Any:
    return operand(jt.Float)

  def Batch(dtype: Any, shape: str = '') -> Any:
    return dtype[jt.Array, f'b {shape}'.strip()]

  # pylint: enable=invalid-name


class ScheduleInfo[Operand](
    typing_extensions.TypedDict,
    total=False,
    extra_items=Time[Operand],
):
  """Schedule evaluation dictionary (open to schedule-specific keys)."""

  time: Time[Operand]
  alpha: Time[Operand]
  sigma: Time[Operand]
  logsnr: Time[Operand]


class TargetInfo[Operand](
    typing_extensions.TypedDict,
    total=False,
    extra_items=Data[Operand],
):
  """Training targets and predictions (open to process-specific keys)."""

  x0: Data[Operand]
  epsilon: Data[Operand]
  score: Data[Operand]
  velocity: Data[Operand]
  v: Data[Operand]


###############################################################################
# MARK: Corruption Process
###############################################################################


class CorruptionSchedule[Operand](Protocol):
  """Protocol for corruption schedules."""

  def evaluate(self, time: Time[Operand]) -> ScheduleInfo[Operand]:
    """Evaluate the schedule for a given time and return schedule info."""
    ...


class CorruptionProcess[Operand](Protocol):
  """Base protocol for all corruption processes (continuous and discrete)."""

  if TYPE_CHECKING:

    @property
    def schedule(self) -> CorruptionSchedule[Operand]:
      ...

  def corrupt(
      self,
      key: PRNGKey,
      x0: Data[Operand],
      time: Time[Operand],
  ) -> tuple[Data[Operand], TargetInfo[Operand]]:
    """Corrupt x0 according to time, and return xt and target predictions."""
    ...

  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: Data[Operand],
  ) -> Data[Operand]:
    """Sample from the invariant distribution."""
    ...

  def convert_predictions(
      self,
      prediction: TargetInfo[Operand],
      xt: Data[Operand],
      time: Time[Operand],
  ) -> TargetInfo[Operand]:
    """Convert raw network predictions to the process target representation."""
    ...


###############################################################################
# MARK: Inference
###############################################################################


class InferenceFn[Operand](Protocol):
  """Protocol for an inference function.

  The `InferenceFn` predicts the clean data representation (e.g. `x0`,
  `velocity`, `logits`) from the noisy input `xt` at `time`, optionally
  incorporating conditioning, guidance, and projection.
  """

  def __call__(
      self,
      time: Time[Operand],
      xt: Data[Operand],
      conditioning: Conditioning | None,
  ) -> TargetInfo[Operand]:
    ...


###############################################################################
# MARK: Training
###############################################################################


class DiffusionLoss[Operand](Protocol):
  """Protocol for per-example diffusion loss functions."""

  def __call__(
      self,
      preds: TargetInfo[Operand],
      targets: TargetInfo[Operand],
      time: Time[Operand],
  ) -> Loss[Operand]:
    """Compute the per-example diffusion loss (without batch averaging)."""
    ...


###############################################################################
# MARK: Sampling
###############################################################################


@flax.struct.dataclass(frozen=True, kw_only=True)
class StepInfo[Operand]:
  """Holds metadata for a diffusion sampling step.

  Attributes:
    step: The step index (or array of step indices prior to scanning).
    time: The time at which the step is evaluated.
    rng: The random number generator key(s) for the step.
  """

  step: int | jax.Array
  time: Time[Operand]
  rng: PRNGKey


@flax.struct.dataclass(frozen=True, kw_only=True)
class DiffusionStep[Operand]:
  """Complete state of the diffusion process at a single sampling step.

  Attributes:
    xt: The noisy data at the current step.
    step_info: The `StepInfo` used to compute the current step.
    aux: Auxiliary data carried by the sampler step (e.g. cached logits).
  """

  xt: Data[Operand]
  step_info: StepInfo[Operand]
  aux: Mapping[str, Any]


class SamplerStep[Operand](Protocol):
  """Protocol defining a single-step diffusion sampling algorithm."""

  def initialize(
      self,
      initial_noise: Data[Operand],
      initial_step_info: StepInfo[Operand],
  ) -> DiffusionStep[Operand]:
    """Initializes the step state (e.g. from pure noise)."""
    ...

  def update(
      self,
      prediction: TargetInfo[Operand],
      current_step: DiffusionStep[Operand],
      next_step_info: StepInfo[Operand],
  ) -> DiffusionStep[Operand]:
    """Performs one step of the sampling process to compute the next state."""
    ...

  def finalize(
      self,
      prediction: TargetInfo[Operand],
      current_step: DiffusionStep[Operand],
      last_step_info: StepInfo[Operand],
  ) -> DiffusionStep[Operand]:
    """Performs the final step to produce the clean output sample."""
    ...


class SampleFn[Operand](Protocol):
  """Protocol for high-level sampling orchestration."""

  def __call__(
      self,
      inference_fn: InferenceFn[Operand],
      rng: PRNGKey,
      initial_noise: Data[Operand],
      conditioning: Conditioning | None = None,
  ) -> tuple[DiffusionStep[Operand], DiffusionStep[Operand] | None]:
    ...
