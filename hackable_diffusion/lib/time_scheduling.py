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
"""

import dataclasses
import typing


import jax.numpy as jnp
import jaxtyping

# Type Aliases
PyTree = jaxtyping.PyTree


class TimeSchedule(typing.Protocol):
  """A protocol defining a time schedule."""

  def eval_schedule(self, num_steps: int) -> PyTree:
    ...


#  _____ ___ __  __ _____      ____   ____ _   _ _____ ____  _   _ _     _____
# |_   _|_ _|  \/  | ____|    / ___| / ___| | | | ____|  _ \| | | | |   | ____|
#   | |  | || |\/| |  _|      \___ \| |   | |_| |  _| | | | | | | | |   |  _|
#   | |  | || |  | | |___      ___) | |___|  _  | |___| |_| | |_| | |___| |___
#   |_| |___|_|  |_|_____|    |____/ \____|_| |_|_____|____/ \___/|_____|_____|
# MARK: Time Schedules
@dataclasses.dataclass(frozen=True, kw_only=True)
class UniformTimeSchedule(TimeSchedule):
  """Creates a schedule with uniformly spaced time steps in [ε, 1-ε]."""

  safety_epsilon: float = 1e-6

  def __post_init__(self):
    if self.safety_epsilon < 0.0 or self.safety_epsilon > 1.0:
      raise ValueError(
          'safety_epsilon must be between 0.0 and 1.0, got'
          f' {self.safety_epsilon}'
      )

  def eval_schedule(self, num_steps: int) -> PyTree:
    start, stop = self.safety_epsilon, 1.0 - self.safety_epsilon
    return jnp.linspace(start, stop, num_steps)
