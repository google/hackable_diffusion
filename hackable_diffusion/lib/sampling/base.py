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

import dataclasses
from typing import Protocol
import flax.struct
from hackable_diffusion.lib import hd_typing
import jax
import kauldron.ktyping as kt


#################################################################################
# MARK: Type Aliases
#################################################################################

Int = hd_typing.Int
PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
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

  step: Int
  time: TimeArray
  rng: PRNGKey


StepInfoTree = PyTree[StepInfo]

# MARK: DiffusionStep Data Structure


@flax.struct.dataclass(frozen=True, kw_only=True)
class DiffusionStep:
  """The complete state of the diffusion process at a single step.

  Attributes:
    xt: The noisy data at the current step.
    conditioning: The conditioning data from the prediction model.
    step_info: The `StepInfo` used to compute the current step.
    aux: Additional data computed by the sampler.
  """

  xt: DataArray
  step_info: StepInfo
  aux: PyTree


DiffusionStepTree = PyTree[DiffusionStep]

################################################################################
# MARK: Protocols
################################################################################


class SamplerStep(Protocol):
  """A protocol defining the diffusion sampling algorithm (e.g., DDIM)."""

  def initialize(
      self,
      initial_noise: DataTree,
      initial_step_info: StepInfoTree,
  ) -> DiffusionStepTree:
    """Initializes the `DiffusionStep` (e.g. from pure noise)."""
    ...

  def update(
      self,
      prediction: TargetInfoTree,
      current_step: DiffusionStep,
      next_step_info: StepInfoTree,
  ) -> DiffusionStepTree:
    """Performs one step of the sampling process to compute the next state."""
    ...

  def finalize(
      self,
      prediction: TargetInfoTree,
      current_step: DiffusionStep,
      last_step_info: StepInfoTree,
  ) -> DiffusionStepTree:
    """Performs the final step to produce the clean output sample."""
    ...
