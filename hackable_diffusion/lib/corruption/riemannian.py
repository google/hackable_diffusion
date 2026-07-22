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
from typing import Any

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption import schedules
import kauldron.ktyping as kt

PRNGKey = hd_typing.PRNGKey
DataArray = hd_typing.DataArray
TimeArray = hd_typing.TimeArray
TargetInfo = hd_typing.TargetInfo


@dataclasses.dataclass(kw_only=True, frozen=True)
class RiemannianProcess(hd_api.CorruptionProcess):
  """Riemannian Flow Matching corruption process.

  This is based on https://arxiv.org/abs/2302.03660.

  Given a schedule with interpolation parameter alpha(t):
    x_t = geodesic(x_1, x_0, alpha(t))
    target = alpha'(t) * velocity(x_1, x_0, alpha(t))
  """

  manifold: manifolds.Manifold
  schedule: schedules.RiemannianSchedule

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Sample from the base distribution (uniform) on the manifold."""
    return self.manifold.random_uniform(key, data_spec.shape)

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> tuple[DataArray, TargetInfo]:  # pyrefly: ignore[not-a-type]
    x1 = self.sample_from_invariant(key, data_spec=x0)

    # Evaluate schedule: alpha(t) is the geodesic interpolation parameter.
    alpha_t = jax_helpers.bcast_right(self.schedule.alpha(time), x0.ndim)
    alpha_dot_t = jax_helpers.bcast_right(self.schedule.alpha_dot(time), x0.ndim)

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

  @kt.typechecked
  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfo:
    """Convert predictions to velocity parameterization."""
    if 'velocity' in prediction:
      return prediction
    raise NotImplementedError(
        'Only velocity prediction is supported for RFM currently.'
    )

  @kt.typechecked
  def get_schedule_info(self, time: TimeArray) -> dict[str, Any]:  # pyrefly: ignore[not-a-type]
    return self.schedule.evaluate(time)
