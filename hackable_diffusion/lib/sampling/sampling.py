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

import dataclasses
from typing import Protocol
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.inference import base as inference_base
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
TimeTree = hd_typing.TimeTree

DiffusionStepTree = base.DiffusionStepTree
SamplerStep = base.SamplerStep
StepInfoTree = base.StepInfoTree

InferenceFn = inference_base.InferenceFn
TimeSchedule = time_scheduling.TimeSchedule

################################################################################
# MARK: Protocols
################################################################################


class SampleFn(Protocol):
  """A protocol for a sampling function."""

  def __call__(
      self,
      inference_fn: InferenceFn,
      rng: PRNGKey,
      initial_noise: DataTree,
      conditioning: Conditioning,
  ) -> tuple[DiffusionStepTree, DiffusionStepTree]:
    ...


################################################################################
# MARK: Helper functions
################################################################################


def _split_pytree(full_pytree: PyTree) -> tuple[PyTree, PyTree, PyTree]:
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


def _is_diffusion_leaf(x: PyTree) -> bool:
  """Returns True if the leaf is a DiffusionStep."""
  return isinstance(x, base.DiffusionStep)


def _get_input_inference_fn(
    step_carry: DiffusionStepTree,
) -> tuple[DataTree, TimeTree]:
  """Returns the input to the inference function for a given step."""
  xt = jax.tree.map(
      lambda x: x.xt,
      step_carry,
      is_leaf=_is_diffusion_leaf,
  )
  time = jax.tree.map(
      lambda x: x.step_info.time,
      step_carry,
      is_leaf=_is_diffusion_leaf,
  )
  return xt, time


################################################################################
# MARK: Sampling loop
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiffusionSampler(SampleFn):
  """Returns a sampling function.

  Attributes:
    time_schedule: Defines the sequence of time steps for the process.
    stepper: The sampling algorithm (e.g., DDIM) that updates the state.
    num_steps: The total number of denoising steps.
  """

  time_schedule: TimeSchedule
  stepper: SamplerStep
  num_steps: int

  @typechecked
  def __call__(
      self,
      inference_fn: InferenceFn,
      rng: PRNGKey,
      initial_noise: DataTree,
      conditioning: Conditioning | None = None,
  ) -> tuple[DiffusionStepTree, DiffusionStepTree]:
    """Performs a full reverse diffusion sampling loop for a single sample.

    This function orchestrates the denoising process, starting from an initial
    (usually noisy) state and iteratively refining it.

    Args:
      inference_fn: The trained model used to make predictions at each step.
      rng: A JAX random key for any stochastic operations.
      initial_noise: The starting PyTree, typically containing Gaussian noise.
      conditioning: The conditioning.

    Returns:
      A tuple containing:
        - The final `DiffusionStepTree` of the sampling process.
        - A `DiffusionStepTree` PyTree containing the full trajectory of all
        steps.
    """
    if self.num_steps < 2:
      raise ValueError(
          f'Number of steps must be at least 2, got {self.num_steps}.'
      )

    all_step_infos = self.time_schedule.all_step_infos(
        rng, self.num_steps, initial_noise
    )

    first_step_info, next_step_infos, last_step_info = _split_pytree(
        all_step_infos
    )

    first_step = self.stepper.initialize(
        initial_noise,
        first_step_info,
    )

    def scan_body(step_carry: DiffusionStepTree, next_step_info: StepInfoTree):
      xt, time = _get_input_inference_fn(step_carry)
      prediction = inference_fn(
          xt=xt,
          conditioning=conditioning,
          time=time,
      )
      next_step = self.stepper.update(
          prediction,
          step_carry,
          next_step_info,
      )
      return next_step, next_step  # ('carryover', 'accumulated')

    before_last_step, intermediate_steps = jax.lax.scan(
        scan_body, first_step, next_step_infos
    )

    xt, time = _get_input_inference_fn(before_last_step)
    last_prediction = inference_fn(
        xt=xt,
        conditioning=conditioning,
        time=time,
    )

    last_step = self.stepper.finalize(
        last_prediction,
        before_last_step,
        last_step_info,
    )

    all_steps = _concat_pytree(first_step, intermediate_steps, last_step)
    return last_step, all_steps
