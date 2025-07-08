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

"""Actual implementation of the sampling steps.

This module proposes various implementations but they all have in common
the core logic:

* An `initialize` function that takes a starting state and returns the
  first step of the diffusion process.
* An `update` function that takes the current state and returns the next step.
* A `finalize` function that takes the last state and returns the final
  state.

At every step, the update function takes the current state and returns the next
state. The update is also in charge of computing other auxiliary informations
such as volatility, drifts, etc.

The `PredictionModel is also called within the step and converted into the
relevant representation, for instance score, velocity, etc.
"""

import dataclasses

from hackable_diffusion.lib import denoising
from hackable_diffusion.lib import noise_scheduling
from hackable_diffusion.lib import prediction_modeling

import jax
import jax.numpy as jnp
import jaxtyping


# Type Aliases
Float = jaxtyping.Float
PyTree = jaxtyping.PyTree
Array = jaxtyping.Array

PredictionModel = denoising.PredictionModel
DiffusionStep = denoising.DiffusionStep
StepInfo = denoising.StepInfo

GaussianNoiseSchedule = noise_scheduling.GaussianNoiseSchedule


#  ____  ____  _____      ____ _____ _____ ____
# / ___||  _ \| ____|    / ___|_   _| ____|  _ \
# \___ \| | | |  _|      \___ \ | | |  _| | |_) |
#  ___) | |_| | |___      ___) || | | |___|  __/
# |____/|____/|_____|    |____/ |_| |_____|_|
# MARK: SDE Step
@dataclasses.dataclass(frozen=True, kw_only=True)
class SdeStep(denoising.SamplerStep):
  """Stochastic Differential Equation (SDE) sampling."""

  noise_schedule: GaussianNoiseSchedule
  churn: float

  def initialize(
      self,
      predictor: PredictionModel,
      initial_noise: PyTree,
      conditioning: PyTree,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    initial_prediction = predictor.predict(
        xt=initial_noise,
        conditioning=conditioning,
        t=initial_step_info['t'],
        rng=initial_step_info['rng'],
    )
    return DiffusionStep(
        xt=initial_noise,
        prediction=initial_prediction,
        conditioning=conditioning,
        step_info=initial_step_info,
        aux=dict(
            f=jnp.nan, g=jnp.nan,
            mean=jnp.full(initial_noise.shape, jnp.nan),
            volatility=jnp.nan,
        ),
    )

  def update(
      self,
      predictor: PredictionModel,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    current_step_info = current_step['step_info']
    xt = current_step['xt']
    f = self.noise_schedule.f(current_step_info['t'])
    g = self.noise_schedule.g(current_step_info['t'])

    prediction = predictor.predict(
        xt=xt,
        conditioning=current_step['conditioning'],
        t=current_step_info['t'],
        rng=current_step_info['rng'],
    )
    score = prediction_modeling.get_score(
        process=self.noise_schedule,
        prediction=prediction,
        t=current_step_info['t'],
        xt=xt,
    )

    dt = next_step_info['t'] - current_step_info['t']
    z = jax.random.normal(
        key=next_step_info['rng'],
        shape=score.shape,
    )

    delta = -f * xt + 0.5 * jnp.square(g) * (1 + self.churn**2) * score
    mean = xt + delta * dt
    volatility = jnp.sqrt(dt) * g * self.churn

    new_xt = mean + volatility * z

    return DiffusionStep(
        xt=new_xt,
        prediction=prediction,
        conditioning=current_step['conditioning'],
        step_info=next_step_info,
        aux=dict(f=f, g=g, mean=mean, volatility=volatility),
        )

  def finalize(
      self,
      predictor: PredictionModel,
      current_step: DiffusionStep,
      last_step_info: StepInfo) -> DiffusionStep:
    # TODO(ccrepy): Verify the actual implementation.
    return self.update(predictor, current_step, last_step_info)
