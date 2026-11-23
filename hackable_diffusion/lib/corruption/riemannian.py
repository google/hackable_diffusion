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

"""Riemannian Flow Matching corruption process."""

import dataclasses
from typing import Protocol, TypeAlias

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib import manifolds
import jax.numpy as jnp

Batch: TypeAlias = hd_api.Batch
PRNGKey: TypeAlias = hd_api.PRNGKey
typechecked = jax_helpers.typechecked


################################################################################
# MARK: Riemannian Schedules
################################################################################


class RiemannianSchedule(hd_api.CorruptionSchedule[Batch], Protocol):
  """Protocol for Riemannian schedules.

  Controls the geodesic interpolation via alpha(t):
    x_t = geodesic(x_1, x_0, alpha(t))
    v_t = alpha'(t) * velocity(x_1, x_0, alpha(t))

  Subclasses must implement `alpha`.
  """

  def alpha(self, time: hd_api.Time[Batch]) -> hd_api.Time[Batch]:
    """The geodesic interpolation parameter at time t."""
    ...

  def alpha_dot(self, time: hd_api.Time[Batch]) -> hd_api.Time[Batch]:
    """Time derivative of alpha. Defaults to autodiff."""
    return jax_helpers.egrad(self.alpha)(time)

  @typechecked
  def evaluate(self, time: hd_api.Time[Batch]) -> hd_api.ScheduleInfo[Batch]:
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

  @typechecked
  def alpha(self, time: hd_api.Time[Batch]) -> hd_api.Time[Batch]:
    return 1.0 - time

  @typechecked
  def alpha_dot(self, time: hd_api.Time[Batch]) -> hd_api.Time[Batch]:
    return -jnp.ones_like(time)


@dataclasses.dataclass(kw_only=True, frozen=True)
class RiemannianProcess(hd_api.CorruptionProcess[Batch]):
  """Riemannian Flow Matching corruption process.

  This is based on https://arxiv.org/abs/2302.03660.

  Given a schedule with interpolation parameter alpha(t):
    x_t = geodesic(x_1, x_0, alpha(t))
    target = alpha'(t) * velocity(x_1, x_0, alpha(t))
  """

  manifold: manifolds.Manifold
  schedule: RiemannianSchedule

  @typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: hd_api.Data[Batch],
  ) -> hd_api.Data[Batch]:
    """Sample from the base distribution (uniform) on the manifold."""
    return self.manifold.random_uniform(key, data_spec.shape)

  @typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: hd_api.Data[Batch],
      time: hd_api.Time[Batch],
  ) -> tuple[hd_api.Data[Batch], hd_api.TargetInfo[Batch]]:
    x1 = self.sample_from_invariant(key, data_spec=x0)

    # Evaluate schedule: alpha(t) is the geodesic interpolation parameter.
    alpha_t = self.schedule.alpha(time)
    alpha_dot_t = self.schedule.alpha_dot(time)

    # x_t = geodesic(x1, x0, alpha(t)).
    xt = manifolds.geodesic(self.manifold, x=x1, y=x0, t=alpha_t)

    # By chain rule: d/dt x_t = alpha'(t) * velocity(x1, x0, alpha(t)).
    vel = alpha_dot_t * self.manifold.velocity(x=x1, y=x0, t=alpha_t)

    target_info = {
        'x0': x0,
        'x1': x1,
        'velocity': vel,
    }

    return xt, target_info

  @typechecked
  def convert_predictions(
      self,
      prediction: hd_api.TargetInfo[Batch],
      xt: hd_api.Data[Batch],
      time: hd_api.Time[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    """Convert predictions to velocity parameterization."""
    del xt, time
    if 'velocity' in prediction:
      return prediction
    raise NotImplementedError(
        'Only velocity prediction is supported for RFM currently.'
    )
