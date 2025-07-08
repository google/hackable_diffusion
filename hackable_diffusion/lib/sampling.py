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

"""This module provides the core logic for running the reverse diffusion (sampling).

It defines a sampling loop that orchestrates three key components:
1. A time schedule for the denoising steps.
2. A prediction model (typically a U-Net) to estimate a denoising operation.
3. A sampler step (e.g., DDIM, DDPM) to update the sample at each step.
"""

import typing

from hackable_diffusion.lib import denoising
from hackable_diffusion.lib import noise_scheduling
from hackable_diffusion.lib import time_scheduling

import jax
import jax.numpy as jnp
import jaxtyping


# Type Aliases
Tuple = typing.Tuple
PyTree = jaxtyping.PyTree
PRNGKeyArray = jaxtyping.PRNGKeyArray


PredictionModel = denoising.PredictionModel
SamplerStep = denoising.SamplerStep
GaussianNoiseSchedule = noise_scheduling.GaussianNoiseSchedule
TimeSchedule = time_scheduling.TimeSchedule


DiffusionStep = denoising.DiffusionStep
StepInfo = denoising.StepInfo


def _split_pytree(full_pytree: PyTree) -> typing.Tuple[PyTree, PyTree, PyTree]:
  """Splits a PyTree into first, middle, and last slices of each leaf."""
  return (
      jax.tree_util.tree_map(lambda x: x[0], full_pytree),
      jax.tree_util.tree_map(lambda x: x[1:-1], full_pytree),
      jax.tree_util.tree_map(lambda x: x[-1], full_pytree),
  )


def _concat_pytree(
    first: PyTree, intermediates: PyTree, last: PyTree
) -> PyTree:
  """Concatenates first, middle, and last slices of each leaf into a single PyTree."""
  def concat_leaf(first_, intermediates_, last_):
    return jnp.concatenate([
        jnp.expand_dims(first_, 0),
        intermediates_,
        jnp.expand_dims(last_, 0),
    ])

  return jax.tree.map(concat_leaf, first, intermediates, last)


def sample_one(
    time_schedule: TimeSchedule,
    predictor: PredictionModel,
    stepper: SamplerStep,
    initial_noise: PyTree,
    conditioning: PyTree,
    num_step: int,
    rng: PRNGKeyArray,
) -> Tuple[DiffusionStep, DiffusionStep]:
  """Performs a full reverse diffusion sampling loop for a single sample.

  This function orchestrates the denoising process, starting from an initial
  (usually noisy) state and iteratively refining it.

  Args:
    time_schedule: Defines the sequence of time steps for the process.
    predictor: The trained model used to make predictions at each step.
    stepper: The sampling algorithm (e.g., DDIM) that updates the state.
    initial_noise: The starting PyTree, typically containing Gaussian noise.
    conditioning: The conditioning.
    num_step: The total number of denoising steps.
    rng: A JAX random key for any stochastic operations.

  Returns:
    A tuple containing:
      - The final `DiffusionStep` of the sampling process.
      - A `DiffusionStep` PyTree containing the full trajectory of all steps.
  """
  all_step_infos = dict(
      step=jnp.arange(num_step),
      t=time_schedule.eval_schedule(num_step),
      rng=jax.random.split(rng, num_step),
  )

  first_step_info, next_step_infos, last_step_info = _split_pytree(
      all_step_infos
  )

  first_step = stepper.initialize(
      predictor, initial_noise, conditioning, first_step_info
  )

  def scan_body(step_carry: DiffusionStep, next_step_info: StepInfo):
    next_step = stepper.update(predictor, step_carry, next_step_info)
    return next_step, next_step  # ('carryover', 'accumulated')

  before_last_step, intermediate_steps = jax.lax.scan(
      scan_body, first_step, next_step_infos
  )

  last_step = stepper.finalize(predictor, before_last_step, last_step_info)

  all_steps = _concat_pytree(first_step, intermediate_steps, last_step)
  return last_step, all_steps


def sample_batch(
    time_schedule: TimeSchedule,
    predictor: PredictionModel,
    stepper: SamplerStep,
    initial_noise_batch: PyTree,
    conditioning_batch: PyTree,
    num_step: int,
    rng: PRNGKeyArray,
) -> Tuple[DiffusionStep, DiffusionStep]:
  """Performs the diffusion sampling process one a batch."""
  batch_size = 8

  def _sample_one(initial_noise, conditioning, rng):
    return sample_one(
        time_schedule,
        predictor,
        stepper,
        initial_noise,
        conditioning,
        num_step,
        rng,
    )

  rng_batch = jax.random.split(rng, batch_size)
  return jax.vmap(_sample_one)(
      initial_noise_batch, conditioning_batch, rng_batch
  )
