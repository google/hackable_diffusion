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

"""Diffusion Loss functions."""

import dataclasses
from typing import Protocol
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree

LossOutputTree = hd_typing.LossOutputTree
TimeArray = hd_typing.TimeArray
TimeTree = hd_typing.TimeTree
TargetInfoTree = hd_typing.TargetInfoTree

Schedule = schedules.Schedule

################################################################################
# MARK: Protocols
################################################################################


class WeightFn(Protocol):
  """A schedule-dependent loss-weight term."""

  def __call__(
      self,
      schedule: Schedule,
      preds: TargetInfoTree,
      targets: TargetInfoTree,
      time: TimeArray,
  ) -> TimeArray:
    pass


class DiffusionLoss(Protocol):

  def __call__(
      self,
      preds: TargetInfoTree,
      targets: TargetInfoTree,
      time: TimeTree,
  ) -> LossOutputTree:
    """Compute the diffusion loss (no averaging).

    Args:
      preds: Prediction dict from the model. Contains one or more of the
        prediction types (e.g. x0, epsilon, score, velocity, v for Gaussian).
      targets: Target dict containing one or more of the prediction type keys.
      time: Time array used for noise computation.

    Returns:
      The loss is returned in the batched format, i.e., if `preds` contained
      tensors of shape `[B, ...]`, the returned loss will have shape `[B,]`. To
      get the scalar loss, use `jnp.mean(loss)`. The loss is returned before the
      averaging to allow for other operations such as masking out loss values
      afterwards.
    """


################################################################################
# MARK: Nested wrappers
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedDiffusionLoss(DiffusionLoss):
  """Wrapper for a pytree of noise schedules that is mapped over the data.

  Enables using different noise schedules for different input modalities.
  E.g. a gaussian schedule for the image and a categorical schedule for the
  labels.
  """

  losses: PyTree[DiffusionLoss]

  @typechecked
  def __call__(
      self,
      preds: TargetInfoTree,
      targets: TargetInfoTree,
      time: TimeTree,
  ) -> LossOutputTree:
    return jax.tree.map(
        lambda loss, target, pred, t: loss(
            preds=pred,
            targets=target,
            time=t,
        ),
        self.losses,
        targets,
        preds,
        time,
    )
