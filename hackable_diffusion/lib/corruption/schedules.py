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

"""Schedules used in corruption processes."""

import dataclasses
import math
from typing import Protocol

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

TimeArray = hd_typing.TimeArray

################################################################################
# MARK: Protocols
################################################################################


class Schedule(Protocol):

  def evaluate(self, time: TimeArray) -> dict[str, TimeArray]:  # pyrefly: ignore[not-a-type]
    """Evaluate the schedule for a given time. Return a dictionary of info."""


################################################################################
# MARK: Base Schedules
################################################################################


class GaussianSchedule(Schedule, Protocol):
  """Protocol for Gaussian schedules (alpha, sigma, logsnr)."""

  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """The alpha parameter for xt = alpha * x0 + sigma * epsilon."""
    ...

  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """The sigma parameter for xt = alpha * x0 + sigma * epsilon."""
    ...

  @kt.typechecked
  def logsnr(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """The log signal-to-noise ratio at time t."""
    return 2.0 * (jnp.log(self.alpha(time)) - jnp.log(self.sigma(time)))

  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """The inverse of the logsnr, i.e., inverse_logsnr(logsnr(t))=t."""
    ...

  @kt.typechecked
  def f(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jax_helpers.egrad(self.alpha)(time) / self.alpha(time)

  @kt.typechecked
  def g(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return self.sigma(time) * jnp.sqrt(-jax_helpers.egrad(self.logsnr)(time))

  @kt.typechecked
  def evaluate(self, time: TimeArray) -> dict[str, TimeArray]:  # pyrefly: ignore[not-a-type]
    return {
        'time': time,
        'alpha': self.alpha(time),
        'sigma': self.sigma(time),
        'logsnr': self.logsnr(time),
    }


class DiscreteSchedule(Schedule, Protocol):
  """Protocol for discrete schedules (just an alpha)."""

  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """The probability of keeping the original value."""
    ...

  @kt.typechecked
  def evaluate(self, time: TimeArray) -> dict[str, TimeArray]:  # pyrefly: ignore[not-a-type]
    return {
        'time': time,
        'alpha': self.alpha(time),
    }


################################################################################
# MARK: Simplicial Schedules
################################################################################

SimplicialSchedule = DiscreteSchedule


################################################################################
# MARK: Riemannian Schedules
################################################################################


class RiemannianSchedule(Schedule, Protocol):
  """Protocol for Riemannian schedules.

  Controls the geodesic interpolation via alpha(t):
    x_t = geodesic(x_1, x_0, alpha(t))
    v_t = alpha'(t) * velocity(x_1, x_0, alpha(t))

  Subclasses must implement `alpha`.
  """

  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """The geodesic interpolation parameter at time t."""
    ...

  def alpha_dot(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Time derivative of alpha. Defaults to autodiff."""
    return jax_helpers.egrad(self.alpha)(time)

  @kt.typechecked
  def evaluate(self, time: TimeArray) -> dict[str, TimeArray]:  # pyrefly: ignore[not-a-type]
    return {
        'time': time,
        'alpha': self.alpha(time),
        'alpha_dot': self.alpha_dot(time),
    }


class LinearRiemannianSchedule(RiemannianSchedule):
  """Linear Riemannian schedule: alpha(t) = 1.0 - t.

  This is the standard flow matching schedule where the geodesic interpolation
  parameter equals time directly.
  Note that contrary to the original Riemannian Flow Matching, we assume that at
  time t=0, the process is close to the data distribution, and at time t=1,
  the process is close to the target distribution.
  Hence, we use alpha(t) = 1.0 - t, and alpha_dot(t) = -1.0, instead of
  alpha(t) = t, and alpha_dot(t) = 1.0.
  """

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return 1.0 - time

  @kt.typechecked
  def alpha_dot(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return -jnp.ones_like(time)


################################################################################
# MARK: Gaussian Schedules
################################################################################


class RFSchedule(GaussianSchedule):
  """Rectified Flow schedule."""

  @kt.typechecked
  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jax.nn.sigmoid(-0.5 * logsnr)

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return 1.0 - time

  @kt.typechecked
  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return time


class CosineSchedule(GaussianSchedule):
  """Cosine diffusion schedule."""

  @kt.typechecked
  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return (2 / jnp.pi) * jnp.arctan(jnp.exp(-0.5 * logsnr))

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.cos(0.5 * jnp.pi * time)

  @kt.typechecked
  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.sin(0.5 * jnp.pi * time)


class InverseCosineSchedule(GaussianSchedule):
  """Inverse Cosine diffusion schedule from https://arxiv.org/abs/2311.17901."""

  @jax_helpers.CustomGradient
  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[bad-override, not-a-type]
    """Shift and scale the inverse cosine function."""
    return jnp.sqrt(self._v(time) / jnp.pi)

  @alpha.derivative
  def alpha_der(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    v = self._v(time)
    return -1.0 / (2.0 * jnp.sqrt(jnp.pi * v) * self._sqrt_t_m_t2(time))

  @jax_helpers.CustomGradient
  @kt.typechecked
  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[bad-override, not-a-type]
    return jnp.sqrt(1.0 - jnp.square(self.alpha(time)))

  @sigma.derivative
  def sigma_der(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    t = time
    v = self._v(t)
    denom = 2.0 * jnp.sqrt(jnp.pi * (jnp.pi - v)) * self._sqrt_t_m_t2(t)
    return 1.0 / denom

  @kt.typechecked
  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return (jnp.cos(jnp.pi * jax.scipy.special.expit(logsnr)) + 1.0) * 0.5

  @kt.typechecked
  def logsnr(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    u = self._v(time) / jnp.pi
    return jax.scipy.special.logit(u)

  @kt.typechecked
  def f(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return -1.0 / (2.0 * self._v(time) * self._sqrt_t_m_t2(time))

  @kt.typechecked
  def g(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    t = time
    denominator = jnp.sqrt(self._v(t) * self._sqrt_t_m_t2(t))
    return 1.0 / denominator

  def _v(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """A common term in many of the functions."""
    return jnp.arccos(2.0 * time - 1)

  def _sqrt_t_m_t2(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """A common term in many of the functions."""
    return jnp.sqrt(time - jnp.square(time))


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearDiffusionSchedule(GaussianSchedule):
  """Linear diffusion schedule, see https://arxiv.org/abs/2011.13456 (eq 33)."""

  beta_min: float = 0.1
  beta_max: float = 20

  @property
  def beta_diff(self) -> float:
    return self.beta_max - self.beta_min

  @jax_helpers.CustomGradient
  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[bad-override, not-a-type]
    return jnp.exp(
        -0.5 * (self.beta_min * time + 0.5 * jnp.square(time) * self.beta_diff)
    )

  @alpha.derivative
  def alpha_der(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    r = -0.5 * (time * self.beta_diff + self.beta_min)
    return self.alpha(time) * r

  @jax_helpers.CustomGradient
  @kt.typechecked
  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[bad-override, not-a-type]
    return jnp.sqrt(1.0 - jnp.square(self.alpha(time)))

  @sigma.derivative
  def sigma_der(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return -self.alpha_der(time) * self.alpha(time) / self.sigma(time)  # pyrefly: ignore[bad-argument-count]

  @kt.typechecked
  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Inverse of logsnr."""

    # The quadratic eqn is: 0.5 (bmax-bmin)t^2 + bmin t - log(1 + 1/snr) = 0
    inversesnr = jnp.exp(-logsnr)
    delta = self.beta_min**2 + 2 * self.beta_diff * jnp.log1p(inversesnr)
    numerator = -self.beta_min + jnp.sqrt(delta)
    denominator = self.beta_diff
    return numerator / denominator

  @kt.typechecked
  def logsnr(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return -jnp.log(
        jnp.expm1(
            self.beta_min * time + 0.5 * jnp.square(time) * self.beta_diff
        )
    )

  @kt.typechecked
  def f(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return -0.5 * (self.beta_min + time * self.beta_diff)

  @kt.typechecked
  def g(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.sqrt(self.beta_min + time * self.beta_diff)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GeometricSchedule(GaussianSchedule):
  """Geometric schedule (similar to VESDE).

  Used in https://arxiv.org/abs/2402.06121 (see F.1)
  """

  sigma_min: float
  sigma_max: float

  @property
  def log_ratio(self) -> float:
    return math.log(self.sigma_max) - math.log(self.sigma_min)

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.ones_like(time)

  @jax_helpers.CustomGradient
  @kt.typechecked
  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[bad-override, not-a-type]
    return self.sigma_min * jnp.exp(time * self.log_ratio)

  @sigma.derivative
  def sigma_der(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return self.log_ratio * self.sigma(time)

  @kt.typechecked
  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    numerator = logsnr + 2 * jnp.log(self.sigma_min)
    denominator = 2 * self.log_ratio
    return -numerator / denominator

  @kt.typechecked
  def logsnr(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return -2.0 * jnp.log(self.sigma_min) - 2.0 * time * self.log_ratio

  @kt.typechecked
  def g(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.sqrt(2.0 * self.sigma(time) * self.sigma_der(time))  # pyrefly: ignore[bad-argument-count]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShiftedSchedule(GaussianSchedule):
  """Shifted schedule.

  This schedule takes any GaussianSchedule and shifts it following
  https://arxiv.org/abs/2410.19324.

  Attributes:
    original_schedule: The original GaussianSchedule.
    target_resolution: The target resolution (corresponds to `image_resolution`
      in Simple Diffusion).
    base_resolution: The base resolution (corresponds to `noise_resolution` in
      Simple Diffusion). The bias is computed as
      -2*(log(target_resolution/base_resolution)). This is in logSNR space.
    logsnr_max: The maximum logSNR for the original schedule.
    logsnr_min: The minimum logSNR for the original schedule.
  """

  original_schedule: GaussianSchedule
  target_resolution: int
  base_resolution: int

  logsnr_max: float = 15.0
  logsnr_min: float = -15.0

  @property
  def logsnr_shift(self):
    """The bias to be added to the logsnr."""
    return -2.0 * (
        jnp.log(self.target_resolution) - jnp.log(self.base_resolution)
    )

  @property
  def tmin(self):
    """The minimum time for the original schedule."""
    return self.original_schedule.inverse_logsnr(jnp.array([self.logsnr_max]))

  @property
  def tmax(self):
    """The maximum time for the original schedule."""
    return self.original_schedule.inverse_logsnr(jnp.array([self.logsnr_min]))

  @kt.typechecked
  def logsnr(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Map time to logSNR of the shifted schedule."""
    rescaled_time = time * (self.tmax - self.tmin) + self.tmin
    rescaled_logsnr = self.original_schedule.logsnr(rescaled_time)
    rescaled_shifted_logsnr = rescaled_logsnr + self.logsnr_shift
    return rescaled_shifted_logsnr

  @kt.typechecked
  def inverse_logsnr(self, logsnr: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Map logSNR of the shifted schedule to time."""
    shifted_logsnr = logsnr - self.logsnr_shift
    shifted_time = self.original_schedule.inverse_logsnr(shifted_logsnr)
    rescaled_shifted_time = (shifted_time - self.tmin) / (self.tmax - self.tmin)
    return rescaled_shifted_time

  @kt.typechecked
  def time_change(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Time change from shifted to original process.

    For a given input time `t`, finds logSNR(t) of the shifted schedule, and
    then computes the time in the original schedule that corresponds to
    logSNR(t).


    Args:
      time: Input time.

    Returns:
      The time in the original schedule that corresponds to logSNR in the
      shifted schedule.
    """
    return self.original_schedule.inverse_logsnr(self.logsnr(time))

  @kt.typechecked
  def inverse_time_change(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Time change from original to shifted process."""
    return self.inverse_logsnr(self.original_schedule.logsnr(time))

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return self.original_schedule.alpha(self.time_change(time))

  @kt.typechecked
  def sigma(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return self.original_schedule.sigma(self.time_change(time))


################################################################################
# MARK: Discrete Schedules
################################################################################
class LinearDiscreteSchedule(DiscreteSchedule):
  """Linear schedule for alpha for discrete corruption processes."""

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return 1.0 - time


class CosineDiscreteSchedule(DiscreteSchedule):
  """Cosine schedule for alpha for discrete corruption processes."""

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.cos(0.5 * jnp.pi * time)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SquareCosineDiscreteSchedule(DiscreteSchedule):
  """Square cosine schedule for alpha for discrete corruption processes.

  This is used in DiGress https://arxiv.org/abs/2209.14734. This is a discrete
  counterpart of the cosine schedule used in the continuous version, see
  https://arxiv.org/abs/2102.09672. Common value is s=0.008 in DiGress.

  Attributes:
    s: shift parameter.
  """

  s: float = 0.0

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    out = jnp.square(jnp.cos(0.5 * jnp.pi * (time + self.s) / (1.0 + self.s)))
    return out / jnp.square(jnp.cos(0.5 * jnp.pi * self.s / (1.0 + self.s)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class GeometricDiscreteSchedule(DiscreteSchedule):
  """Geometric schedule for alpha for discrete corruption processes.

  Used for discrete diffusion by https://openreview.net/forum?id=71mqtQdKB9
  """

  beta_min: float
  beta_max: float

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return jnp.exp(-self.beta_min ** (1 - time) * self.beta_max**time)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolynomialDiscreteSchedule(DiscreteSchedule):
  """Polynomial schedule for alpha for discrete corruption processes."""

  degree: float = 1.0

  @kt.typechecked
  def alpha(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    return 1 - time**self.degree
