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

"""Defines noise schedules for continuous-time Gaussian diffusion models."""

import abc
import dataclasses
import typing
from typing import Tuple

import jax
import jax.numpy as jnp

import jaxtyping


# Type Aliases
Float = jaxtyping.Float
PyTree = jaxtyping.PyTree

ScalarFn = typing.Callable[[Float], Float]


# Helper functions
def _manual_gradient_definition(
    *, value_fn: ScalarFn, derivative_fn: ScalarFn, x: Float
) -> Float:
  """Defines a JAX custom gradient for a given scalar function.

  This helper allows replacing JAX's default autodifferentiation with a manually
  provided, it is useful to provide numerical stability.

  Args:
    value_fn: The function to compute the forward pass value, f(x).
    derivative_fn: The function to compute the derivative, f'(x).
    x: The input point at which to evaluate the function and its gradient.

  Returns:
    The result of value_fn(x), with its gradient defined by derivative_fn.
  """

  @jax.custom_gradient
  def _custom_gradient_wrapper(x_val: Float) -> Float:
    """The wrapper that applies the custom gradient logic."""

    def grad_fn(dt: Float) -> Float:
      # The gradient is the derivative of the function scaled by the incoming
      # gradient from the next operation in the chain (dt).
      return dt * derivative_fn(x_val)

    return value_fn(x_val), grad_fn

  return _custom_gradient_wrapper(x)


#  ____    _    ____  _____      ____  ____   ___   ____ _____ ____ ____
# | __ )  / \  / ___|| ____|    |  _ \|  _ \ / _ \ / ___| ____/ ___/ ___|
# |  _ \ / _ \ \___ \|  _|      | |_) | |_) | | | | |   |  _| \___ \___ \
# | |_) / ___ \ ___) | |___     |  __/|  _ <| |_| | |___| |___ ___) |__) |
# |____/_/   \_\____/|_____|    |_|   |_| \_\\___/ \____|_____|____/____/
# MARK: Base Process
@dataclasses.dataclass(frozen=True, kw_only=True)
class GaussianNoiseSchedule(abc.ABC):
  """Abstract base class for Gaussian noise schedules."""

  @abc.abstractmethod
  def alpha(self, t: Float) -> Float:
    """The signal coefficient. Must be implemented by subclasses."""
    raise NotImplementedError()

  @abc.abstractmethod
  def sigma(self, t: Float) -> Float:
    """The noise coefficient. Must be implemented by subclasses."""
    raise NotImplementedError()

  def logsnr(self, t: Float) -> Float:

    def value_fn(t: Float):
      return 2. * jnp.log(self.alpha(t)) - jnp.log(self.sigma(t))

    def derivative_fn(t: Float):
      return 2. * (
          jax.grad(self.alpha)(t) * jnp.reciprocal(self.alpha(t))
          - jax.grad(self.sigma)(t) * jnp.reciprocal(self.sigma(t))
          )

    return _manual_gradient_definition(
        value_fn=value_fn,
        derivative_fn=derivative_fn,
        x=t,
    )

  def f(self, t: Float) -> Float:
    return jax.grad(self.alpha)(t) * jnp.reciprocal(self.alpha(t))

  def g(self, t: Float) -> Float:
    return self.sigma(t) * jnp.sqrt(-jax.grad(self.logsnr)(t))


#  ____  ____   ___   ____ _____ ____ ____    ___ __  __ ____  _
# |  _ \|  _ \ / _ \ / ___| ____/ ___/ ___|  |_ _|  \/  |  _ \| |
# | |_) | |_) | | | | |   |  _| \___ \___ \   | || |\/| | |_) | |
# |  __/|  _ <| |_| | |___| |___ ___) |__) |  | || |  | |  __/| |___
# |_|   |_| \_\\___/ \____|_____|____/____/  |___|_|  |_|_|   |_____|
# MARK: Process Implementations
@dataclasses.dataclass(frozen=True, kw_only=True)
class RFSchedule(GaussianNoiseSchedule):

  def alpha(self, t: Float) -> Float:
    return 1. - t

  def sigma(self, t: Float) -> Float:
    return t


@dataclasses.dataclass(frozen=True, kw_only=True)
class CosineSchedule(GaussianNoiseSchedule):

  def alpha(self, t: Float) -> Float:
    return jnp.cos(0.5 * jnp.pi * t)

  def sigma(self, t: Float) -> Float:
    return jnp.sin(0.5 * jnp.pi * t)
