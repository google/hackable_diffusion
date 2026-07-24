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

"""Gaussian corruption processes.

*** CORRUPTION PROCESS ***

This module defines the Gaussian diffusion process.
A diffusion process is defined as follows

  X(t) = α(t) X(0) + σ(t) Z, (1)

with Z ~ N(0, Id) and X(0) ~ Pi (Pi is the target data distribution).
This is implemented in "add_noise".

It can be the case that the quantity of interest (either in the sampler or in
the conditioning of the network) depends not on the time but on the logSNR. The
logSNR is defined as

  logSNR(t) = 2 log(α(t) / σ(t)).

Importantly, the logSNR is a non-increasing quantity in all considered
schedules (it goes from +∞ at t=0 to -∞ at t=1).

The associated forward process is given by

  dX(t) = f(t) X(t) dt + g(t) dB(t), (2)

where B(t) is a Brownian motion and X(0) ~ Pi. The quantities f(t) and g(t) are
defined as follows

  f(t) = log(α(t))',
  g(t)**2 = 2 α(t) σ(t) (σ(t)/α(t))',
  g(t) = σ(t)sqrt(-logSNR_der(t)).

Note that in that case, we not only have (1) but also the more general formula

  X(t) = (α(t)/α(s)) X(s) + (σ(t)^2 - (σ(s)α(t)/α(s))^2)^(1/2) Z, (3)

with Z ~ N(0, Id) and X(u), s <= t, solution of (2) started at X(s).
In addition, we have the following useful identities

  X(s) = r(t) X(t) + α(s) (1 - r2(t)) X(0) + σ(s) (1 - r2(t))^(1/2) Z, (4)

where Z ~ N(0, Id) and we have

  r(t) = (α(t)/α(s)) (σ(s)/σ(t))^2
  r2(t) = (α(t)/α(s))^2 (σ(s)/σ(t))^2

Example: in the special case of the cosine schedule, we have

  α(t) = cos(0.5 π t),
  σ(t) = sin(0.5 π t),
  f(t) = -0.5 π tan(0.5 π t),
  g(t)**2 = π tan(0.5 π t).

Note that we *never* need to implement the forward process in the differential
formulation (2) and only use (1). However, the differential formulation (2)
gives us the backward process Y(t) such that

  dY(t) = [-f(τ) Y(t) + g(τ)**2 s(τ, Y(t))] dt + g(τ) dB(t), (5)

where Y(0) ~ N(0, Id), τ=1-t and s is the score (the logarithmic gradient of the
density of X(t)). The formulation (5) is what is discretised in practice to get
the generative model.

One can check that letting s -> t in (4) we recover (5).

Note that (5) corresponds to a *stochastic* sampler but *deterministic*
samplers are also available. Indeed, (5) has the same marginal distributions as

  dY(t) = [-f(τ) Y(t) + 0.5 (1 + ε**2) g(τ)**2 s(τ, Y(t))] + ε g(τ) dB(t), (6)

where Y(0) ~ N(0, Id), s is the score, τ=1-t and 0 <= ε <= 1. In the case where
ε=1 we recover the Stochastic Differential Equation (SDE) (6) while with ε=0 we
recover the following Ordinary Differential Equation (ODE)

  dY(t) = [-f(τ) Y(t) + 0.5 g(τ)**2 s(τ, Y(t))] dt, (7)

where Y(0) ~ N(0, Id), s is the score, τ=1-t. Note that (7) is *deterministic*.
This framework also encompasses flow matching (stochastic interpolant) as a
special choice of schedule. We refer to the following papers for more details:
* https://arxiv.org/abs/2011.13456
* https://arxiv.org/abs/2303.08797

Writing things in terms of velocity, we can define a forward process with the
same distribution as (1) with

  dX(t) = v(t, X(t)) dt, (8)

with X(0) ~ Pi and v(t, x) = E[grad(α)(t) X(0) + grad(σ)(t) Z | X(t) = x]. The
quantity v is called the velocity. This gives us a *deterministic* backward
process associated with (8)

  dY(t) = -v(τ, Y(t)) dt, (9)

where Y(0) ~ N(0, Id), v is the velocity and τ=1-t. As in (6), we can define a
stochastic sampler by adding noise to (7). In that case, we get

  dY(t) = [-v(τ, Y(t)) + 0.5 * ε**2 * s(τ, Y(t))] dt + ε dB(t), (10)

where Y(0) ~ N(0, Id), s is the score, v is the velocity, τ=1-t and 0 <= ε <= 1.
Importantly, the SDE (10) and the SDE (6) are equivalent, but correspond to
different points of view on the conception of diffusion.

*** PREDICTION PARAMETERIZATION ***

For Gaussian processes we consider the following prediction parameterizations:

* x0
* epsilon
* score
* velocity
* v

Starting from (1), this amounts to predicting:

* x0 -- X(0)
* epsilon -- Z
* score -- -Z / σ(t)
* velocity -- α(t)' X(0) + σ(t)' Z
* v -- α(t) Z - σ(t) X(0)

The velocity target can be found in the Flow Matching
(https://arxiv.org/abs/2210.02747), Rectified Flow
(https://arxiv.org/abs/2209.03003) and
Stochastic Interpolant (https://arxiv.org/abs/2303.08797) papers.
The v-prediction was first introduced in "Progressive Distillation for Fast
Sampling of Diffusion Models" (https://arxiv.org/abs/2202.00512).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
import immutabledict
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey

DataArray = hd_typing.DataArray
TimeArray = hd_typing.TimeArray
TargetInfo = hd_typing.TargetInfo

CorruptionProcess = hd_api.CorruptionProcess


################################################################################
# MARK: Gaussian Schedules
################################################################################


class GaussianSchedule(hd_api.CorruptionSchedule, Protocol):
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
# MARK: GaussianProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianProcess:
  """Gaussian corruption process.

  Takes the following form with epsilon ~ N(0, 1):
  xt = alpha(t) * x0 + sigma(t) * epsilon

  The schedule parameters are alpha and sigma.
  The corresponding target / prediction parameterizations are: x0, epsilon,
  score, velocity and v.
  """

  schedule: GaussianSchedule

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Sample from the invariant distribution."""
    return jax.random.normal(key, shape=data_spec.shape)

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> tuple[DataArray, TargetInfo]:  # pyrefly: ignore[not-a-type]
    epsilon = self.sample_from_invariant(key, data_spec=x0)

    time = jax_helpers.bcast_right(time, x0.ndim)
    alpha, sigma, alpha_der, sigma_der = self._get_alpha_sigma_and_der(time)

    xt = alpha * x0 + sigma * epsilon

    target_info = {
        'x0': x0,
        'epsilon': epsilon,
        'score': -epsilon / sigma,
        'velocity': alpha_der * x0 + sigma_der * epsilon,
        'v': alpha * epsilon - sigma * x0,
    }

    return xt, target_info

  @kt.typechecked
  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfo:
    if len(prediction) != 1:
      raise KeyError(
          f'Exactly one prediction is required. Got: {prediction.keys()=}'
      )

    source_type, source_value = next(iter(prediction.items()))
    converters = CONVERTERS[source_type]
    time = jax_helpers.bcast_right(time, xt.ndim)
    alpha, sigma, alpha_der, sigma_der = self._get_alpha_sigma_and_der(time)

    return {
        pred_type: converter(
            source_value,
            xt=xt,
            alpha=alpha,
            sigma=sigma,
            alpha_der=alpha_der,
            sigma_der=sigma_der,
        )
        for pred_type, converter in converters.items()
    }

  @kt.typechecked
  def _get_alpha_sigma_and_der(
      self, time: TimeArray  # pyrefly: ignore[not-a-type]
  ) -> tuple[TimeArray, TimeArray, TimeArray, TimeArray]:  # pyrefly: ignore[not-a-type]
    """Get the alpha, sigma and their derivatives for the given time."""
    alpha = self.schedule.alpha(time)
    sigma = self.schedule.sigma(time)
    alpha_der = jax_helpers.egrad(self.schedule.alpha)(time)
    sigma_der = jax_helpers.egrad(self.schedule.sigma)(time)
    return alpha, sigma, alpha_der, sigma_der


################################################################################
# MARK: Conversion Functions
################################################################################

################################################################################
# MARK: Convert From x0
################################################################################


def x0_to_epsilon(x0, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  return (xt - alpha * x0) / sigma


def x0_to_score(x0, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  return (alpha * x0 - xt) / jnp.square(sigma)


def x0_to_velocity(x0, xt, alpha, sigma, alpha_der, sigma_der):
  # Intermediate epsilon = (xt - alpha * x0) / sigma
  return alpha_der * x0 + sigma_der * ((xt - alpha * x0) / sigma)


def x0_to_v(x0, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  # Intermediate epsilon = (xt - alpha * x0) / sigma
  return alpha * ((xt - alpha * x0) / sigma) - sigma * x0


################################################################################
# MARK: Convert From Epsilon
################################################################################


def epsilon_to_x0(epsilon, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  return (xt - sigma * epsilon) / alpha


def epsilon_to_score(epsilon, xt, alpha, sigma, alpha_der, sigma_der):
  del xt, alpha, alpha_der, sigma_der  # Unused
  return -epsilon / sigma


def epsilon_to_velocity(epsilon, xt, alpha, sigma, alpha_der, sigma_der):
  # Intermediate x0 = (xt - sigma * epsilon) / alpha
  return alpha_der * ((xt - sigma * epsilon) / alpha) + sigma_der * epsilon


def epsilon_to_v(epsilon, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  # Intermediate x0 = (xt - sigma * epsilon) / alpha
  return alpha * epsilon - sigma * ((xt - sigma * epsilon) / alpha)


################################################################################
# MARK: Convert From Score
################################################################################


def score_to_x0(score, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  return (xt + jnp.square(sigma) * score) / alpha


def score_to_epsilon(score, xt, alpha, sigma, alpha_der, sigma_der):
  del xt, alpha, alpha_der, sigma_der  # Unused
  return -score * sigma


def score_to_velocity(score, xt, alpha, sigma, alpha_der, sigma_der):
  # Intermediate x0 = (xt + jnp.square(sigma) * score) / alpha
  # Intermediate epsilon = -score * sigma
  return alpha_der * ((xt + jnp.square(sigma) * score) / alpha) + sigma_der * (
      -score * sigma
  )


def score_to_v(score, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  # Intermediate x0 = (xt + jnp.square(sigma) * score) / alpha
  # Intermediate epsilon = -score * sigma
  return alpha * (-score * sigma) - sigma * (
      (xt + jnp.square(sigma) * score) / alpha
  )


################################################################################
# MARK: Convert From Velocity
################################################################################


def velocity_to_x0(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  # Solves:
  # velocity = alpha_der * x0 + sigma_der * epsilon
  # xt = alpha * x0 + sigma * epsilon
  # For x0.
  common_denominator = alpha_der * sigma - sigma_der * alpha
  return (velocity * sigma - sigma_der * xt) / common_denominator


def velocity_to_epsilon(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  # Solves the same system for epsilon.
  common_denominator = alpha_der * sigma - sigma_der * alpha
  return (alpha_der * xt - alpha * velocity) / common_denominator


def velocity_to_score(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  # score = -epsilon / sigma
  # Intermediate epsilon_numerator = alpha_der * xt - alpha * velocity
  # Intermediate epsilon_denominator = alpha_der * sigma - sigma_der * alpha
  return (alpha * velocity - alpha_der * xt) / (
      sigma * (alpha_der * sigma - sigma_der * alpha)
  )


def velocity_to_v(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  # v = alpha * epsilon - sigma * x0
  # Inlined expressions for x0 and epsilon from velocity:
  common_denominator = alpha_der * sigma - sigma_der * alpha
  # x0_num = velocity * sigma - sigma_der * xt
  # eps_num = alpha_der * xt - alpha * velocity
  # v = (alpha * eps_num - sigma * x0_num) / common_denominator
  numerator = (alpha * alpha_der + sigma * sigma_der) * xt - (
      jnp.square(alpha) + jnp.square(sigma)
  ) * velocity
  return numerator / common_denominator


################################################################################
# MARK: Convert From v
################################################################################


def v_to_x0(v, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  # Solves:
  # v = alpha * epsilon - sigma * x0
  # xt = alpha * x0 + sigma * epsilon
  # For x0.
  common_denominator = jnp.square(alpha) + jnp.square(sigma)
  return (alpha * xt - sigma * v) / common_denominator


def v_to_epsilon(v, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  # Solves the same system for epsilon.
  common_denominator = jnp.square(alpha) + jnp.square(sigma)
  return (sigma * xt + alpha * v) / common_denominator


def v_to_score(v, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der  # Unused
  # score = -epsilon / sigma
  # Inlined expression for epsilon from v:
  # epsilon_val = (sigma * xt + alpha * v) /
  #               (jnp.square(alpha) + jnp.square(sigma))
  # return -epsilon_val / sigma
  return -(sigma * xt + alpha * v) / (
      sigma * (jnp.square(alpha) + jnp.square(sigma))
  )


def v_to_velocity(v, xt, alpha, sigma, alpha_der, sigma_der):
  # velocity = alpha_der * x0 + sigma_der * epsilon
  # Inlined expressions for x0 and epsilon from v:
  common_denominator = jnp.square(alpha) + jnp.square(sigma)
  # x0_num = alpha * xt - sigma * v
  # eps_num = sigma * xt + alpha * v
  # velocity = (alpha_der * x0_num + sigma_der * eps_num) / common_denominator
  numerator = (alpha_der * alpha + sigma_der * sigma) * xt + (
      sigma_der * alpha - alpha_der * sigma
  ) * v
  return numerator / common_denominator


################################################################################
# MARK: Helpers
################################################################################


def _identity(y, xt, alpha, sigma, alpha_der, sigma_der):
  del xt, alpha, sigma, alpha_der, sigma_der  # Unused
  return y


CONVERTERS = immutabledict.immutabledict({
    'x0': {
        'x0': _identity,
        'epsilon': x0_to_epsilon,
        'score': x0_to_score,
        'velocity': x0_to_velocity,
        'v': x0_to_v,
    },
    'epsilon': {
        'x0': epsilon_to_x0,
        'epsilon': _identity,
        'score': epsilon_to_score,
        'velocity': epsilon_to_velocity,
        'v': epsilon_to_v,
    },
    'score': {
        'x0': score_to_x0,
        'epsilon': score_to_epsilon,
        'score': _identity,
        'velocity': score_to_velocity,
        'v': score_to_v,
    },
    'velocity': {
        'x0': velocity_to_x0,
        'epsilon': velocity_to_epsilon,
        'score': velocity_to_score,
        'velocity': _identity,
        'v': velocity_to_v,
    },
    'v': {
        'x0': v_to_x0,
        'epsilon': v_to_epsilon,
        'score': v_to_score,
        'velocity': v_to_velocity,
        'v': _identity,
    },
})
