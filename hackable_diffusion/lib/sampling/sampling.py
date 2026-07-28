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

"""This module provides the core logic for running the reverse diffusion (sampling).

It defines a sampling loop that orchestrates three key components:
1. A time schedule for the denoising steps.
2. A prediction model (typically a U-Net) to estimate a denoising operation.
3. A sampler step (e.g., DDIM, DDPM) to update the sample at each step.
"""

import dataclasses
from typing import Protocol
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.sampling import diffusion_early_stopping
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
DataArray = hd_typing.DataArray
TimeTree = hd_typing.TimeTree
TimeArray = hd_typing.TimeArray

DiffusionStep = hd_api.DiffusionStep
StepInfo = hd_api.StepInfo
SamplerStep = hd_api.SamplerStep

DiffusionStepTree = PyTree[DiffusionStep]


InferenceFn = hd_api.InferenceFn
TimeSchedule = time_scheduling.TimeSchedule
DiffusionEarlyStoppingFn = diffusion_early_stopping.DiffusionEarlyStoppingFn

################################################################################
# MARK: Protocols
################################################################################


class UpdateConditioningFn(Protocol):
  """Protocol for updating conditioning during the sampling loop.

  This allows injecting step-dependent information back into the conditioning
  dict between sampling steps (e.g. self-conditioning logits from the
  previous prediction).
  """

  def __call__(
      self,
      conditioning: Conditioning,
      step_carry: PyTree[DiffusionStep],  # pyrefly: ignore[not-a-type]
  ) -> Conditioning:
    """Update conditioning based on the current diffusion step.

    Args:
      conditioning: The current conditioning dict.
      step_carry: The current diffusion step state.

    Returns:
      The updated conditioning dict.
    """
    ...


################################################################################
# MARK: Helper Functions
################################################################################


def _split_pytree(full_pytree: PyTree) -> tuple[PyTree, PyTree, PyTree]:  # pyrefly: ignore[not-a-type]
  """Splits a PyTree into first, middle, and last slices of each leaf."""
  return (
      jax.tree_util.tree_map(lambda x: x[0], full_pytree),
      jax.tree_util.tree_map(lambda x: x[1:-1], full_pytree),
      jax.tree_util.tree_map(lambda x: x[-1], full_pytree),
  )


def _concat_pytree(
    first: PyTree, intermediates: PyTree, last: PyTree  # pyrefly: ignore[not-a-type]
) -> PyTree:  # pyrefly: ignore[not-a-type]
  """Concatenates first, middle, and last slices of each leaf into a single PyTree."""

  def concat_leaf(first_, intermediates_, last_):
    return jnp.concatenate([
        jnp.expand_dims(first_, 0),
        intermediates_,
        jnp.expand_dims(last_, 0),
    ])

  return jax.tree.map(concat_leaf, first, intermediates, last)


def _is_diffusion_leaf(x: PyTree) -> bool:  # pyrefly: ignore[not-a-type]
  """Returns True if the leaf is a DiffusionStep."""
  return isinstance(x, hd_api.DiffusionStep)


def _get_input_inference_fn(
    step_carry: DiffusionStepTree,  # pyrefly: ignore[not-a-type]
) -> tuple[DataTree, TimeTree]:  # pyrefly: ignore[not-a-type]
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


def _index_pytree(pytree: PyTree, idx: int) -> PyTree:  # pyrefly: ignore[not-a-type]
  """Indexes into the leading axis of every leaf in a PyTree."""
  return jax.tree.map(lambda x: x[idx], pytree)


def _freeze_done_elements(
    new_step: PyTree[DiffusionStep],  # pyrefly: ignore[not-a-type]
    old_step: PyTree[DiffusionStep],  # pyrefly: ignore[not-a-type]
    done: jax.Array,
) -> PyTree[DiffusionStep]:  # pyrefly: ignore[not-a-type]
  """Keeps old values for batch elements that are already done.

  For each leaf array with a leading batch dimension, elements where
  ``done[b]`` is True are replaced with the corresponding values from
  ``old_step``.  Non-batched leaves (e.g. scalar step counters, RNG keys)
  are passed through from ``new_step`` unchanged.

  Args:
    new_step: The freshly computed PyTree[DiffusionStep].
    old_step: The previous PyTree[DiffusionStep] (carry from last iteration).
    done: Bool array of shape ``[batch_size]``.

  Returns:
    A PyTree[DiffusionStep] where done elements are frozen.
  """
  batch_size = done.shape[0]

  def _select_and_replace(new_leaf, old_leaf):
    # Only freeze leaves whose leading dim matches batch_size.
    if new_leaf.ndim == 0 or new_leaf.shape[0] != batch_size:
      return new_leaf
    done_bcast = done.reshape((-1,) + (1,) * (new_leaf.ndim - 1))
    return jnp.where(done_bcast, old_leaf, new_leaf)

  return jax.tree.map(_select_and_replace, new_step, old_step)


################################################################################
# MARK: Sampling loop
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiffusionSampler(hd_api.SampleFn):
  """Returns a sampling function.

  Attributes:
    time_schedule: Defines the sequence of time steps for the process.
    stepper: The sampling algorithm (e.g., DDIM) that updates the state.
    num_steps: The total number of denoising steps.
    store_trajectory: If True (default), returns the full trajectory of all
      intermediate steps. If False, only returns the final step and None for the
      trajectory, significantly reducing memory usage for high-resolution or
      many-step sampling.
    update_conditioning_fn: An optional function to update the conditioning at
      each step.
  """

  time_schedule: TimeSchedule
  stepper: SamplerStep
  num_steps: int
  store_trajectory: bool = True
  update_conditioning_fn: UpdateConditioningFn | None = None

  @kt.typechecked
  def __call__(
      self,
      inference_fn: InferenceFn,
      rng: PRNGKey,
      initial_noise: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None = None,
  ) -> tuple[DiffusionStepTree, DiffusionStepTree | None]:  # pyrefly: ignore[not-a-type]
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
        - The final `PyTree[DiffusionStep]` of the sampling process.
        - A `PyTree[DiffusionStep]` PyTree containing the full trajectory of all
        steps, or None if `store_trajectory` is False.
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

    def _step(step_carry: PyTree[DiffusionStep], next_step_info: PyTree[StepInfo]):  # pyrefly: ignore[not-a-type]
      xt, time = _get_input_inference_fn(step_carry)
      updated_conditioning = conditioning
      if self.update_conditioning_fn is not None:
        updated_conditioning = self.update_conditioning_fn(
            conditioning, step_carry  # pyrefly: ignore[bad-argument-type]
        )
      prediction = inference_fn(
          xt=xt,
          conditioning=updated_conditioning,
          time=time,
      )
      next_step = self.stepper.update(
          prediction,
          step_carry,
          next_step_info,
      )
      return next_step

    def scan_body(step_carry, next_step_info):
      next_step = _step(step_carry, next_step_info)
      return next_step, next_step  # ('carryover', 'accumulated')

    before_last_step, intermediate_steps = jax.lax.scan(
        scan_body, first_step, next_step_infos
    )

    xt, time = _get_input_inference_fn(before_last_step)
    last_conditioning = conditioning
    if self.update_conditioning_fn is not None:
      last_conditioning = self.update_conditioning_fn(
          conditioning, before_last_step  # pyrefly: ignore[bad-argument-type]
      )
    last_prediction = inference_fn(
        xt=xt,
        conditioning=last_conditioning,
        time=time,
    )

    last_step = self.stepper.finalize(
        last_prediction,
        before_last_step,
        last_step_info,
    )

    if self.store_trajectory:
      all_steps = _concat_pytree(first_step, intermediate_steps, last_step)
    else:
      all_steps = None
    return last_step, all_steps


################################################################################
# MARK: DiffusionSamplerWithEarlyStopping
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiffusionSamplerWithEarlyStopping(hd_api.SampleFn):
  """Samples via reverse diffusion using ``jax.lax.while_loop``.

  This sampler replaces ``jax.lax.scan`` with ``jax.lax.while_loop`` which
  supports early stopping. After the whole while loop, it does one last finalize
  step. Operates on the arrays and does not support tree structures at the
  moment.

  Attributes:
    time_schedule: Defines the sequence of time steps for the process.
    stepper: The sampling algorithm (e.g., DDIM) that updates the state.
    num_steps: The total number of denoising steps.
    update_conditioning_fn: An optional function to update the conditioning at
      each step.
    early_stopping_fn: An optional function that determines when to stop
      denoising early, per batch element.  Defaults to ``NoEarlyStop`` (run all
      steps).
    use_lax_while_loop: An optional flag to choose between lax while loop or
      pythonic one.
  """

  time_schedule: TimeSchedule
  stepper: SamplerStep
  num_steps: int
  update_conditioning_fn: UpdateConditioningFn | None = None
  early_stopping_fn: DiffusionEarlyStoppingFn = (
      diffusion_early_stopping.DiffusionNoEarlyStopFn()
  )
  use_lax_while_loop: bool = True

  @kt.typechecked
  def __call__(
      self,
      inference_fn: InferenceFn,
      rng: PRNGKey,
      initial_noise: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None = None,
  ) -> tuple[DiffusionStep, DiffusionStep | None]:
    """Performs a full reverse diffusion sampling loop for a single sample.

    Uses ``jax.lax.while_loop`` instead of ``jax.lax.scan``.

    The ``last_step.step_info.step`` field records the step index at which
    the loop terminated (useful for logging how many denoising steps were
    actually executed when early stopping is active).

    Args:
      inference_fn: The trained model used to make predictions at each step.
      rng: A JAX random key for any stochastic operations.
      initial_noise: The starting noise, typically containing Gaussian noise.
      conditioning: The conditioning.

    Returns:
      A tuple containing:
        - The last ``DiffusionStep`` of the sampling process.  Its
          ``step_info.step`` field records the actual number of update steps
          executed.
        - ``None`` (trajectory storage is not supported).

    Raises:
      ValueError: If ``num_steps < 1``.
    """
    if self.num_steps < 1:
      raise ValueError(
          f'Number of steps must be at least 1, got {self.num_steps}.'
      )

    # Pre-compute all step infos: shape [num_steps, ...] for each leaf.
    all_step_infos = self.time_schedule.all_step_infos(
        rng, self.num_steps, initial_noise
    )
    # Initialize from the first step info.
    first_step_info = _index_pytree(all_step_infos, 0)
    first_step = self.stepper.initialize(initial_noise, first_step_info)

    # The total number of intermediate update steps excluding finalize.
    num_intermediate_steps = self.num_steps - 1

    # done: per-element bool tracking which batch elements have converged.
    batch_size = initial_noise.shape[0]
    done = jnp.zeros(batch_size, dtype=jnp.bool_)

    # Carry: (step_carry, step_counter, done)
    init_carry = (first_step, jnp.int32(0), done)

    def _cond_fn(carry):
      _, step, done = carry
      within_budget = step < num_intermediate_steps
      any_active = ~jnp.all(done)
      return jnp.logical_and(within_budget, any_active)

    def _body_fn(carry):
      step_carry, step, done = carry

      # Index +1 because index 0 is the init step info.
      next_step_info = _index_pytree(all_step_infos, step + 1)

      xt, time = _get_input_inference_fn(step_carry)
      updated_conditioning = conditioning
      if self.update_conditioning_fn is not None:
        updated_conditioning = self.update_conditioning_fn(
            conditioning, step_carry  # pyrefly: ignore[bad-argument-type]
        )
      prediction = inference_fn(
          xt=xt,
          conditioning=updated_conditioning,
          time=time,
      )
      next_step = self.stepper.update(
          prediction,
          step_carry,
          next_step_info,
      )

      # Check early stopping (per batch element).
      new_done = done | self.early_stopping_fn.should_stop(
          step=step,
          current_step=next_step,
          previous_step=step_carry,
      )

      # Freeze carry for already-done elements.
      next_step = _freeze_done_elements(next_step, step_carry, done)

      return (next_step, step + 1, new_done)

    if self.use_lax_while_loop:
      before_last_step, steps_executed, done = jax.lax.while_loop(
          _cond_fn, _body_fn, init_carry
      )
    else:
      carry = init_carry
      while _cond_fn(carry):
        carry = _body_fn(carry)
      before_last_step, steps_executed, done = carry

    # Finalize: run the last step.
    last_step_info = _index_pytree(all_step_infos, num_intermediate_steps)
    xt, time = _get_input_inference_fn(before_last_step)
    last_conditioning = conditioning
    if self.update_conditioning_fn is not None:
      last_conditioning = self.update_conditioning_fn(
          conditioning, before_last_step  # pyrefly: ignore[bad-argument-type]
      )
    last_prediction = inference_fn(
        xt=xt,
        conditioning=last_conditioning,
        time=time,
    )

    last_step = self.stepper.finalize(
        last_prediction,
        before_last_step,
        last_step_info,
    )

    # Freeze already-done elements so finalize doesn't overwrite them.
    last_step = _freeze_done_elements(last_step, before_last_step, done)

    # Record the actual number of steps executed in step_info.step.
    # steps_executed counts update steps; +1 accounts for finalize.
    actual_total_steps = steps_executed + 1
    last_step = jax.tree.map(
        lambda node: node.replace(
            step_info=node.step_info.replace(step=actual_total_steps)
        ),
        last_step,
        is_leaf=_is_diffusion_leaf,
    )

    return last_step, None
