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

"""Defines protocols and implementations for diffusion time-step schedules.

This module provides a `TimeSchedule` protocol to abstract the discretization
of the `t` in [0, 1] time interval, allowing different scheduling strategies to
be used interchangeably.

IMPORTANT: The timesteps are flipped such that t=0 corresponds to the last time
step and t=1 to the first time step. This in accordance with the notation in
the diffusion literature.
"""

import dataclasses
from typing import Protocol
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.sampling import base as sampling_base
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree

StepInfoTree = sampling_base.StepInfoTree
StepInfo = sampling_base.StepInfo

################################################################################
# MARK: Protocols
################################################################################


class TimeSchedule(Protocol):
  """A protocol defining a time schedule."""

  def all_step_infos(
      self,
      rng: PRNGKey,
      num_steps: int,
      data_spec: DataTree,
  ) -> StepInfoTree:
    """Returns all the step infos for a given number of steps.

    We refer to sampling.py for the more details on the use of all the step
    infos within the sampling loop.

    Args:
      rng: A JAX random key for any stochastic operations.
      num_steps: The number of steps to generate.
      data_spec: A pytree of data arrays, used to determine the shape of the
        time tree.

    Returns:
      A pytree of step infos, where each step info is a dict with the following
      keys:
        * step: The step number.
        * time: The time value.
        * rng: The random number generator key.
    """
    ...


################################################################################
# MARK: Uniform Time Schedule
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class UniformTimeSchedule(TimeSchedule):
  """Creates a schedule with uniformly spaced time steps in [ε, 1-ε]."""

  safety_epsilon: float = 1e-6

  def __post_init__(self):
    if self.safety_epsilon < 0.0 or self.safety_epsilon > 1.0:
      raise ValueError(
          "safety_epsilon must be between 0.0 and 1.0, got"
          f" {self.safety_epsilon}"
      )

  @typechecked
  def all_step_infos(
      self, rng: PRNGKey, num_steps: int, data_spec: DataArray
  ) -> StepInfo:
    bsz, *data_shape = data_spec.shape
    start, stop = 1.0 - self.safety_epsilon, self.safety_epsilon
    steps = jnp.linspace(start, stop, num_steps)
    steps = utils.bcast_right(steps, data_spec.ndim + 1)
    steps = jnp.repeat(steps, bsz, axis=1)

    expected_steps_shape = (
        num_steps,
        bsz,
    ) + (
        1,
    ) * len(data_shape)
    if steps.shape != expected_steps_shape:
      raise ValueError(
          f"Expected steps to have shape {expected_steps_shape}, got"
          f" {steps.shape}"
      )
    return StepInfo(
        step=jnp.arange(num_steps),
        time=steps,
        rng=jax.random.split(rng, num_steps),
    )


################################################################################
# MARK: EDM Time Schedule
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class EDMTimeSchedule(TimeSchedule):
  """Creates a schedule with non-uniformly spaced time steps in [ε, 1-ε].

  The implementation is based on https://arxiv.org/abs/2206.00364.
  First, we compute a uniformly spaced schedule in [ε^(1/rho), (1-ε)^(1/rho)].

  vec = [tmin^(1/rho), ..., tmax^(1/rho)]

  Then, we put this vector into the range [ε, 1-ε] by putting it to the power of
  rho. The final schedule is vec^rho. Note that if rho=1.0, the schedule is
  uniform.
  """

  safety_epsilon: float = 1e-6
  rho: float = 1.0

  def __post_init__(self):
    if self.safety_epsilon < 0.0 or self.safety_epsilon > 1.0:
      raise ValueError(
          "safety_epsilon must be between 0.0 and 1.0, got"
          f" {self.safety_epsilon}"
      )

  @typechecked
  def all_step_infos(
      self, rng: PRNGKey, num_steps: int, data_spec: DataArray
  ) -> StepInfo:
    bsz, *data_shape = data_spec.shape
    start, stop = 1.0 - self.safety_epsilon, self.safety_epsilon
    start_inv_rho = start ** (1.0 / self.rho)
    stop_inv_rho = stop ** (1.0 / self.rho)
    steps = jnp.linspace(start_inv_rho, stop_inv_rho, num_steps)
    steps = steps**self.rho
    steps = utils.bcast_right(steps, data_spec.ndim + 1)
    steps = jnp.repeat(steps, bsz, axis=1)

    expected_steps_shape = (num_steps, bsz) + (1,) * len(data_shape)
    if steps.shape != expected_steps_shape:
      raise ValueError(
          f"Expected steps to have shape {expected_steps_shape}, got"
          f" {steps.shape}"
      )
    return StepInfo(
        step=jnp.arange(num_steps),
        time=steps,
        rng=jax.random.split(rng, num_steps),
    )


################################################################################
# MARK: Nested Time Schedule
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedTimeSchedule(TimeSchedule):
  """Wrapper to support a nested pytree of time schedules.

  The structure of the time schedule should match the structure of the data.

  Usage Example:
    ```
    time_schedule = NestedTimeSchedule(
        time_schedules={
            "image": UniformTimeSchedule(),
            "label": EDMTimeSchedule(rho=2.0),
        }
    )
    ```

  Attributes:
    time_schedules: A pytree of time schedules matching the structure of the
      data.
  """

  time_schedules: PyTree[TimeSchedule]

  @typechecked
  def all_step_infos(
      self,
      rng: PRNGKey,
      num_steps: int,
      data_spec: DataTree,
  ) -> StepInfoTree:
    def _call_schedule(rng, time_schedule, data_spec):
      return time_schedule.all_step_infos(rng, num_steps, data_spec)

    return utils.tree_map_with_key(
        _call_schedule, rng, self.time_schedules, data_spec
    )
