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

# pylint: disable=line-too-long
"""This module defines the core data structures and protocols for a diffusion sampling loop.

It the `PredictionModel` (the neural network) is visible from the `SamplerStep`
(the algorithm, e.g., DDIM).


 The following diagram illustrates the flow of the denoising process:

                             ┌──────────────────┐
                             │ Model / Backbone │
                             └───────────┬──────┘
                                  ▲      │
   ─ ─ ─ ─ ─ ─ ─ ─  ┐     ┌ ─ ─ ─ │ ─ ─ ─│─ ─ ─ ─ ─ ┐     ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─
                    │     │       │      │
                                  │      │          │
                    │ predict(...)│      │ prediction     │
                                  │      │   {'x0: ...}
                    │     │              ▼          │     │
                                 update(...)
                    │     │   ┌─────────────────┐   │     │
                              │                 ▼
                 ┌────────────┴──┐           ┌───────────────┐
                 │ DiffusionStep │           │ DiffusionStep │
                 │      T-1      │           │       T       │
                 └───────────────┘           └───────────────┘
                    │     │                         │     │
  ─ STEP T-1  ─ ─ ─ ┘     └ ─ ─ ─ ─ STEP T  ─ ─ ─ ─ ┘     └ ─ ─ STEP T+1  ─ ─

# TODO(ccrepy): Add stepinfo to the diagram.

 At each step T, the `SamplerStep.update()` calls the `PredictionModel` to
 produce the next `DiffusionStep` from step T-1.

 Each `DiffusionStep` is a complete snapshot of the process at a single point
 in time, acting as a full autoregressive state.
 
"""

import typing

import jaxtyping

# Type Aliases
Float = jaxtyping.Float
Int = jaxtyping.Int
PyTree = jaxtyping.PyTree
Array = jaxtyping.Array
PRNGKeyArray = jaxtyping.PRNGKeyArray


#  ____    _  _____  _      ____ _____ ____  _   _  ____ _____ _   _ ____  _____ ____
# |  _ \  / \|_   _|/ \    / ___|_   _|  _ \| | | |/ ___|_   _| | | |  _ \| ____/ ___|
# | | | |/ _ \ | | / _ \   \___ \ | | | |_) | | | | |     | | | | | | |_) |  _| \___ \
# | |_| / ___ \| |/ ___ \   ___) || | |  _ <| |_| | |___  | | | |_| |  _ <| |___ ___) |
# |____/_/   \_\_/_/   \_\ |____/ |_| |_| \_\\___/ \____| |_|  \___/|_| \_\_____|____/
# MARK: Data Structures
class StepInfo(typing.TypedDict):
  """Holds metadata for the current diffusion step.

  Attributes:
    step: The step number.
    t: The time at which the step is computed.
    rng: The random number generator key.

  All these fields are static and are computed before starting the sampling loop.
  """
  step: Int
  t: PyTree
  rng: PRNGKeyArray


class DiffusionStep(typing.TypedDict):
  """The complete state of the diffusion process at a single step.

  Attributes:
    xt: The noisy data at the current step, same shape as `x0`.
    prediction: The predicted clean data as a dict, `x0` is always populated.
      Unimodal case is `{'x0': Array}` but more general formats are possible.
      Example of PyTree shape for multimodal data and containing `velocity`:
        {
            'img': {'x0': Array, 'velocity': Array}
            'text': {'x0': Array, 'velocity':  Array}
        }
    conditioning: The conditioning data from the prediction model.
    step_info: The `StepInfo` used to compute the current step.
    aux: Additional data computed by the sampler and passed along for debugging
      purposes.
  """
  xt: PyTree
  prediction: PyTree
  conditioning: PyTree
  step_info: StepInfo
  aux: PyTree


#  ____  ____   ___ _____ ___   ____ ___  _     ____
# |  _ \|  _ \ / _ \_   _/ _ \ / ___/ _ \| |   / ___|
# | |_) | |_) | | | || || | | | |  | | | | |   \___ \
# |  __/|  _ <| |_| || || |_| | |__| |_| | |___ ___) |
# |_|   |_| \_\\___/ |_| \___/ \____\___/|_____|____/
# MARK: Protocols
class PredictionModel(typing.Protocol):
  """A protocol for a model that predicts the initial state `x0`."""

  def predict(self, xt: PyTree, conditioning: PyTree, t: PyTree, rng: PRNGKeyArray) -> PyTree:
    """Predicts the clean data `x0` from the noisy input `xt`, possibly with guidance and/or conditioning."""
    ...


class SamplerStep(typing.Protocol):
  """A protocol defining the diffusion sampling algorithm (e.g., DDIM)."""

  def initialize(self, predictor: PredictionModel, initial_noise: PyTree, conditioning: PyTree, initial_step_info: StepInfo) -> DiffusionStep:
    """Initializes the first `DiffusionStep` from a starting state (e.g., pure noise)."""
    ...

  def update(self, predictor: PredictionModel, current_step: DiffusionStep, next_step_info: StepInfo) -> DiffusionStep:
    """Performs one step of the sampling process to compute the next state."""
    ...

  def finalize(self, predictor: PredictionModel, current_step: DiffusionStep, last_step_info: StepInfo) -> DiffusionStep:
    """Performs the final step to produce the clean output sample."""
    ...
