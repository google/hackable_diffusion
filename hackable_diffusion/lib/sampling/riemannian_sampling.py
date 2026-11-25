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

"""Riemannian Flow Matching sampler step."""

import dataclasses
from typing import TypeAlias

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import riemannian

################################################################################
# MARK: Type Aliases
################################################################################

Batch: TypeAlias = hd_api.Batch
typechecked = jax_helpers.typechecked


################################################################################
# MARK: Sampler Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class RiemannianFlowSamplerStep(hd_api.SamplerStep[Batch]):
  """Euler integration on Riemannian manifold for Flow Matching."""

  corruption_process: riemannian.RiemannianProcess

  @typechecked
  def initialize(
      self,
      initial_noise: hd_api.Data[Batch],
      initial_step_info: hd_api.StepInfo[Batch],
  ) -> hd_api.DiffusionStep[Batch]:
    return hd_api.DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={},
    )

  @typechecked
  def update(
      self,
      prediction: hd_api.TargetInfo[Batch],
      current_step: hd_api.DiffusionStep[Batch],
      next_step_info: hd_api.StepInfo[Batch],
  ) -> hd_api.DiffusionStep[Batch]:
    xt = current_step.xt
    t = current_step.step_info.time
    next_t = next_step_info.time
    dt = next_t - t

    v = prediction['velocity']

    # Riemannian Euler integration step. The exponential map generalizes the
    # Euclidean update x_{t+dt} = x_t + dt * v to manifolds.
    next_xt = self.corruption_process.manifold.exp(xt, dt * v)

    return hd_api.DiffusionStep(
        xt=next_xt,
        step_info=next_step_info,
        aux={},
    )

  @typechecked
  def finalize(
      self,
      prediction: hd_api.TargetInfo[Batch],
      current_step: hd_api.DiffusionStep[Batch],
      last_step_info: hd_api.StepInfo[Batch],
  ) -> hd_api.DiffusionStep[Batch]:
    return self.update(prediction, current_step, last_step_info)
