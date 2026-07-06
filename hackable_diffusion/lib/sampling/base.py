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

# pylint: disable=line-too-long,g-docstring-first-line-too-long
"""This module defines the core data structures and protocols for a diffusion sampling loop.

The `InferenceFn` (pure update function) is visible from the `SamplerStep`
(the algorithm, e.g., DDIM).


 The following diagram illustrates the flow of the denoising process:

                             ┌──────────────────┐
                             │ Model / Backbone │
                             └───────────┬──────┘
                                  ▲      │
   ─ ─ ─ ─ ─ ─ ─ ─  ┐     ┌ ─ ─ ─ │ ─ ─ ─│─ ─ ─ ─ ─ ┐     ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─
                    │     │       │      │          |     |
                    |     |       │      │          │     |
                 inference_fn(...)│      │ prediction     │
                    |     |       │      │   {'x0: ...}   |
                    │     │       |      ▼          │     │
                    |     |        update(...)      |     |
                    │     │   ┌─────────────────┐   │     │
                    |     |   │                 ▼   |     |
                 ┌────────────┴──┐           ┌───────────────┐
                 │ DiffusionStep │           │ DiffusionStep │
                 │      T-1      │           │       T       │
                 └───────────────┘           └───────────────┘
                    |     |                         |     |
                 ┌───────────────┐           ┌───────────────┐
                 │    StepInfo   │           │    StepInfo   │
                 │      T-1      │           │       T       │
                 └───────────────┘           └───────────────┘
                    |     |                         |     |
                    │     │                         │     │
  ─ STEP T-1  ─ ─ ─ ┘     └ ─ ─ ─ ─ STEP T  ─ ─ ─ ─ ┘     └ ─ ─ STEP T+1  ─ ─


 At each step T, the `SamplerStep.update()` calls the `InferenceFn` to
 produce the next `DiffusionStep` from step T+1.

 Each `DiffusionStep` is a complete snapshot of the process at a single point
 in time, acting as a full autoregressive state.

 Each `StepInfo` contains the static information needed to compute the current
 step, such as the step number, time, and rng key.

 At the end of the sampling loop for the last step, the `SamplerStep.finalize()`
 is called to produce the final clean output sample.
"""

from typing import Protocol
import flax.struct
from hackable_diffusion.lib import hd_typing

#################################################################################
# MARK: Type Aliases
#################################################################################

Int = hd_typing.Int
PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
Conditioning = hd_typing.Conditioning
TargetInfoTree = hd_typing.TargetInfoTree
TimeArray = hd_typing.TimeArray

#################################################################################
# MARK: StepInfo Data Structure
#################################################################################


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


StepInfoTree = PyTree[StepInfo]

################################################################################
# MARK: DiffusionStep Data Structure
################################################################################


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


DiffusionStepTree = PyTree[DiffusionStep]


################################################################################
# MARK: Protocols
################################################################################


class SamplerStep(Protocol):
  """A protocol defining the diffusion sampling algorithm (e.g., DDIM)."""

  def initialize(
      self,
      initial_noise: DataTree,  # pyrefly: ignore[not-a-type]
      initial_step_info: StepInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> DiffusionStepTree:  # pyrefly: ignore[not-a-type]
    """Initializes the `DiffusionStep` (e.g. from pure noise)."""
    ...

  def update(
      self,
      prediction: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      current_step: DiffusionStep,
      next_step_info: StepInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> DiffusionStepTree:  # pyrefly: ignore[not-a-type]
    """Performs one step of the sampling process to compute the next state."""
    ...

  def finalize(
      self,
      prediction: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      current_step: DiffusionStep,
      last_step_info: StepInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> DiffusionStepTree:  # pyrefly: ignore[not-a-type]
    """Performs the final step to produce the clean output sample."""
    ...


class UpdateConditioningFn(Protocol):
  """Protocol for updating conditioning during the sampling loop.

  This allows injecting step-dependent information back into the conditioning
  dict between sampling steps (e.g. self-conditioning logits from the
  previous prediction).
  """

  def __call__(
      self,
      conditioning: Conditioning,
      step_carry: DiffusionStepTree,  # pyrefly: ignore[not-a-type]
  ) -> Conditioning:
    """Update conditioning based on the current diffusion step.

    Args:
      conditioning: The current conditioning dict.
      step_carry: The current diffusion step state.

    Returns:
      The updated conditioning dict.
    """
    ...
