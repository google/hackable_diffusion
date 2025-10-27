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

The `InferenceFn is also called within the step and converted into the
relevant representation, for instance score, velocity, etc.
"""

import dataclasses

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

GaussianProcess = gaussian.GaussianProcess
TimeSchedule = time_scheduling.TimeSchedule

################################################################################
# MARK: SDE Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class SdeStep(SamplerStep):
  """Stochastic Differential Equation (SDE) sampling.

  Attributes:
    corruption_process: The corruption process to use.
    churn: The churn parameter.
    stochastic_last_step: Whether the last step is stochastic.
  """

  corruption_process: GaussianProcess
  churn: float
  stochastic_last_step: bool = False

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
      stochastic: bool = True,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    f = self.corruption_process.schedule.f(time)
    g = self.corruption_process.schedule.g(time)

    score = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )["score"]

    dt = time - next_time
    z = jax.random.normal(
        key=next_step_info.rng,
        shape=score.shape,
    )

    delta = (
        -f * xt + 0.5 * jnp.square(g) * (1.0 + jnp.square(self.churn)) * score
    )
    mean = xt + delta * dt
    volatility = jnp.sqrt(dt) * g * self.churn

    if stochastic:
      new_xt = mean + volatility * z
    else:
      new_xt = mean

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
        stochastic=self.stochastic_last_step,
    )


################################################################################
# MARK: Adjusted DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class AdjustedDDIMStep(SamplerStep):
  """Adjusted DDIM sampler from https://arxiv.org/pdf/2403.06807.

  We refer to Algorithm 3 of the paper. The strategy used to estimate the
  covariance is the one described page 7, corresponding to 10% of the posterior
  variance if the prior was a factorized Gaussian with variance 0.5.

  Attributes:
    corruption_process: The corruption process to use.
  """

  corruption_process: GaussianProcess

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    alpha = self.corruption_process.schedule.alpha(time)
    sigma = self.corruption_process.schedule.sigma(time)
    next_alpha = self.corruption_process.schedule.alpha(next_time)
    next_sigma = self.corruption_process.schedule.sigma(next_time)

    prediction_dict = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )
    x0 = prediction_dict["x0"]
    epsilon = prediction_dict["epsilon"]

    # Estimation according to page 7 of the paper.
    xt_var = 0.1 / (2 + jnp.square(alpha / sigma))
    # Ajusted DDIM update.
    next_xt_var = jnp.square(next_alpha - alpha * next_sigma / sigma) * xt_var
    norm_epsilon = jnp.mean(jnp.square(epsilon), keepdims=True)
    new_xt = (
        alpha * x0
        + jnp.sqrt(jnp.square(next_sigma) + next_xt_var / norm_epsilon)
        * epsilon
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )


################################################################################
# MARK: DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class DDIMStep(SamplerStep):
  """DDIM sampler from https://arxiv.org/abs/2010.02502.

  stoch_coeff controls the interpolation between DDIM and DDPM:
  stoch_coeff = 0.0 gives (deterministic) DDIM and stoch_coeff = 1.0 gives DDPM.

  Attributes:
    corruption_process: The corruption process to use.
    stoch_coeff: The interpolation parameter between DDIM and DDPM.
    stochastic_last_step: Whether the last step is stochastic.
  """

  corruption_process: GaussianProcess
  stoch_coeff: float
  stochastic_last_step: bool = False

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
      stochastic: bool = True,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    x0 = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )["x0"]
    z = jax.random.normal(key=next_step_info.rng, shape=x0.shape)

    alpha = self.corruption_process.schedule.alpha(time)
    sigma = self.corruption_process.schedule.sigma(time)
    next_alpha = self.corruption_process.schedule.alpha(next_time)
    next_sigma = self.corruption_process.schedule.sigma(next_time)

    # alpha sigma ratios
    r01 = next_sigma / sigma
    r11 = alpha / next_alpha * r01
    r12 = r11 * r01
    r22 = r11 * r11

    # DDIM update
    coeff_xt = self.stoch_coeff * r12 + (1.0 - self.stoch_coeff) * r01
    coeff_x0 = next_alpha * (
        1.0 - self.stoch_coeff * r22 - (1.0 - self.stoch_coeff) * r11
    )
    volatility = next_sigma * jnp.sqrt(
        1.0 - jnp.square(self.stoch_coeff * r11 + (1.0 - self.stoch_coeff))
    )
    new_mean = coeff_xt * xt + coeff_x0 * x0
    if stochastic:
      new_xt = new_mean + volatility * z
    else:
      new_xt = new_mean

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
        stochastic=self.stochastic_last_step,
    )


################################################################################
# MARK: Velocity Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class VelocityStep(SamplerStep):
  """DDIM sampler from https://arxiv.org/abs/2010.02502.

  epsilon controls the interpolation between DDIM and DDPM:
  epsilon = 0.0 gives (deterministic) DDIM and epsilon = 1.0 gives DDPM.
  """

  corruption_process: GaussianProcess
  epsilon: float

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
      stochastic: bool = True,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    g = self.corruption_process.schedule.g(time)

    dt = time - next_time

    prediction_dict = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )
    velocity = prediction_dict["velocity"]
    score = prediction_dict["score"]
    z = jax.random.normal(key=next_step_info.rng, shape=xt.shape)

    delta = -velocity + 0.5 * self.epsilon**2 * g**2 * score
    new_mean = xt + delta * dt
    volatility = jnp.sqrt(dt) * g * self.epsilon

    if stochastic:
      new_xt = new_mean + volatility * z
    else:
      new_xt = new_mean

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
        stochastic=False,
    )


################################################################################
# MARK: Heun Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class HeunStep(SamplerStep):
  """Heun sampler, adapted from https://arxiv.org/abs/2206.00364.

  Importantly, since the Heun sampler is a multi-step sampler and contains one
  internal step, we must ensure that the time schedule used in the sampler has a
  number of steps which is even. The state at each odd step is an **internal**
  state.

  Attributes:
    corruption_process: The corruption process to use.
    time_schedule: The time schedule used.
    num_steps: The total number of steps (including internal steps).
  """

  corruption_process: GaussianProcess
  time_schedule: TimeSchedule
  num_steps: int

  def __post_init__(self):
    if self.num_steps % self.num_internal_steps != 0:
      raise ValueError(
          "The number of steps should be divisible by"
          f" {self.num_internal_steps}, got {self.num_steps}"
      )

  @property
  def num_internal_steps(self):
    return 2

  def _get_all_step_infos(self, rng: PRNGKey, data_spec: DataArray) -> StepInfo:
    return self.time_schedule.all_step_infos(
        rng=rng, num_steps=self.num_steps, data_spec=data_spec
    )

  def _get_specific_step_info(
      self, rng: PRNGKey, data_spec: DataArray, step: int
  ):
    all_step_infos = self._get_all_step_infos(rng=rng, data_spec=data_spec)
    return jax.tree.map(lambda x: x[step], all_step_infos)

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    """Initializes the Heun sampler.

    Note that DiffusionStep contains an auxiliary dictionary which is used to
    keep track of the internal step. This dictionary contains the following
    keys:
      - internal_counter: An integer indicating the current internal step.
      - current_update: The current internal step zero.
      - current_velocity_step_one: The current velocity step one.

    The internal counter is updated at each step and initialized at 0. If the
    internal counter is 0, then we apply `first_step`. Otherwise, we apply the
    `second_step`.

    The `current_update` is the state corresponding to the current
    initial_counter=0. For instance at step 3, this contains the information of
    step 2. This `current_update` is updated in `second_step`.

    The `current_velocity_step_one` is the current intermediate velocity. It is
    updated in `first_step`. Combining `current_velocity_step_one` the velocity
    computed in `second_step` gives the final velocity.

    Args:
      initial_noise: The initial noise.
      initial_step_info: The initial step info.

    Returns:
      The initial step.
    """

    current_update = DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(
            internal_counter=jnp.array(0),
            current_update=current_update,
            current_velocity_step_one=initial_noise,  # placeholder
        ),
    )

  @typechecked
  def first_step(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """First step of the Heun sampler.

    This is the first internal step. It should be called when the internal
    counter is 0.

    Args:
      prediction: The prediction from the corruption process.
      current_step: The current step.
      next_step_info: The next step info.

    Returns:
      The next step.
    """

    current_step_info = current_step.step_info

    xt = current_step.xt

    rng = next_step_info.rng
    # Obtain the next next step info (next external step).
    next_next_step = next_step_info.step + 1
    next_next_step_info = self._get_specific_step_info(
        rng=rng, data_spec=xt, step=next_next_step
    )

    time = current_step_info.time
    next_next_time = next_next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_next_time = utils.bcast_right(next_next_time, xt.ndim)

    # Perform the intermediate step.
    dt = time - next_next_time
    velocity = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )["velocity"]
    new_xt = xt - dt * velocity

    aux = current_step.aux
    # Update the internal counter and current_velocity_step_one.
    internal_counter = jnp.mod(
        aux["internal_counter"] + 1, self.num_internal_steps
    )
    current_velocity_step_one = velocity
    aux.update(
        dict(
            internal_counter=internal_counter,
            current_velocity_step_one=current_velocity_step_one,
        )
    )

    # Note that we output next_next_step_info and not next_step_info.
    return DiffusionStep(
        xt=new_xt,
        step_info=next_next_step_info,
        aux=aux,
    )

  @typechecked
  def second_step(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """First step of the Heun sampler.

    This is the second internal step. It should be called when the internal
    counter is 1.

    Args:
      prediction: The prediction from the corruption process.
      current_step: The current step.
      next_step_info: The next step info.

    Returns:
      The next step.
    """
    aux = current_step.aux
    xt = current_step.xt

    current_update = aux["current_update"]
    old_time = current_update.step_info.time
    next_time = next_step_info.time
    old_time = utils.bcast_right(old_time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    old_velocity = aux["current_velocity_step_one"]
    intermediate_velocity = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=next_time,
    )["velocity"]
    # Note that here we use next_step_info and not current_step_info.
    # This is because `first_step` outputs next_next_step_info.

    # Perform the final step.

    old_xt = current_update.xt
    dt = old_time - next_time
    new_xt = old_xt - dt * (old_velocity + intermediate_velocity) / 2

    # Update the internal counter and the current_update
    internal_counter = jnp.mod(
        aux["internal_counter"] + 1, self.num_internal_steps
    )
    current_update = DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )
    aux.update(
        dict(
            internal_counter=internal_counter,
            current_update=current_update,
        )
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=aux,
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """Update function for the Heun sampler.

    Depending on the internal counter, we call either `first_step` or
    `second_step`.

    Args:
      prediction: The prediction from the corruption process.
      current_step: The current step.
      next_step_info: The next step info.

    Returns:
      The next step.
    """

    return jax.lax.switch(
        current_step.aux["internal_counter"],
        [self.first_step, self.second_step],
        prediction,
        current_step,
        next_step_info,
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )
