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
from hackable_diffusion.lib import random_utils
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import simplicial
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.sampling import base
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

SimplicialProcess = simplicial.SimplicialProcess
SimplicialSchedule = schedules.SimplicialSchedule

################################################################################
# MARK: DDIM Step
################################################################################

# TODO(vdebortoli): Add support for churn.


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimplicialDDIMStep(SamplerStep):
  """This is the simplicial version of the DDIM step."""

  corruption_process: SimplicialProcess

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': initial_noise},
    )
    # `logits` need to be passed in `aux` dictionary to a performance
    # bug when using TPU. Needs to be investigated.

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    log_xt = current_step.xt  # Input is now logits (log-probabilities)

    time = current_step_info.time
    next_time = next_step_info.time

    # Broadcast time to match batch dimensions
    time = utils.bcast_right(time, log_xt.ndim)
    next_time = utils.bcast_right(next_time, log_xt.ndim)
    key = next_step_info.rng

    # Get logits
    logits = self.corruption_process.convert_predictions(
        prediction,
        log_xt,
        time,
    )['logits']

    # Sample hard token
    sample_key, beta_key = jax.random.split(key)
    sample_idx = jax.random.categorical(key=sample_key, logits=logits)
    num_cats = self.corruption_process.process_num_categories
    one_hot_mask = jax.nn.one_hot(sample_idx, num_cats, dtype=log_xt.dtype)
    log_sample_oh = jnp.where(one_hot_mask > 0.5, 0.0, -1e30)

    # Compute Beta shape parameters
    alpha_t = self.corruption_process.schedule.alpha(time)
    alpha_s = self.corruption_process.schedule.alpha(next_time)

    shape_0 = self.corruption_process.temperature / (1.0 - alpha_t)
    shape_1 = self.corruption_process.temperature / (1.0 - alpha_s) - shape_0

    # Broadcasting
    target_shape = log_xt.shape[:-1] + (1,)
    shape_0 = jnp.broadcast_to(shape_0, target_shape)
    shape_1 = jnp.broadcast_to(shape_1, target_shape)

    # Sample from Beta(shape_0, shape_1)
    log_w, log_1_minus_w = random_utils.sample_log_beta_joint(
        beta_key, shape_0, shape_1, shape=shape_0.shape
    )

    term_1 = log_w + log_xt
    term_2 = log_1_minus_w + log_sample_oh

    new_xt = jnp.logaddexp(term_1, term_2)

    return DiffusionStep(
        xt=new_xt,  # Output is robust logits
        step_info=next_step_info,
        aux={'logits': logits},
    )
    # `logits` need to be passed in `aux` dictionary to a performance
    # bug when using TPU. Needs to be investigated.

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
