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

"""Losses for discrete diffusion."""

import dataclasses
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.loss import base
import jax.numpy as jnp
import optax

################################################################################
# MARK: Type Aliases
################################################################################

LossOutput = hd_typing.LossOutput
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

DiscreteSchedule = schedules.DiscreteSchedule

################################################################################
# MARK: General Loss function
################################################################################


@typechecked
def compute_discrete_diffusion_loss(
    preds: TargetInfo,
    targets: TargetInfo,
    time: TimeArray,
    *,
    schedule: DiscreteSchedule | None = None,
    use_mask: bool = False,
    weight_fn: base.WeightFn | None = None,
) -> LossOutput:
  """Compute the discrete diffusion loss."""

  # The last dimension of preds and targets is a vocabulary dimension.
  time = utils.bcast_right(time, targets['x0'].ndim)

  bsz = time.shape[0]

  labels = jnp.squeeze(targets['x0'], axis=-1)
  # Remove trailing dimension of the x0.
  if use_mask:
    mask = jnp.squeeze(targets['mask'], axis=-1)
    # Remove trailing dimension of the mask.
    # Mask is True if xt is not masked. This is to stay consistent with
    # the conditioning mask.
  else:
    mask = jnp.zeros_like(labels, dtype=jnp.bool_)

  neg_xentropy = -optax.softmax_cross_entropy_with_integer_labels(
      logits=preds['logits'],
      labels=labels,
      where=jnp.invert(mask),
  )

  assert neg_xentropy.shape == labels.shape

  reduce_axes = tuple(range(1, neg_xentropy.ndim))
  neg_xentropy = jnp.mean(neg_xentropy, axis=reduce_axes, keepdims=True)
  neg_xentropy = utils.flatten_non_batch_dims(neg_xentropy)

  assert neg_xentropy.shape == (bsz, 1)

  if weight_fn is not None:
    weight = weight_fn(
        schedule=schedule,
        preds=preds,
        targets=targets,
        time=time,
    )
    weight = utils.flatten_non_batch_dims(weight)
  else:
    # No weighting is applied.
    weight = 1.0
  weighted_loss = -1.0 * weight * neg_xentropy

  weighted_loss = jnp.squeeze(weighted_loss, axis=-1)

  assert weighted_loss.shape == (bsz,)
  return weighted_loss


################################################################################
# MARK: Specific Loss functions
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoWeightDiscreteLoss(base.DiffusionLoss):
  """Discrete loss without weight."""

  use_mask: bool = False

  @typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:

    return compute_discrete_diffusion_loss(
        preds=preds,
        targets=targets,
        time=time,
        schedule=None,
        use_mask=self.use_mask,
        weight_fn=None,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MD4Loss(base.DiffusionLoss):
  """MD4 loss as in https://arxiv.org/abs/2406.04329, Eq 5."""

  schedule: DiscreteSchedule
  use_mask: bool = False

  @typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    def _weight_fn(
        schedule: DiscreteSchedule,
        preds: TargetInfo,
        targets: TargetInfo,
        time: TimeArray,
    ) -> TimeArray:
      """Weight function for the MD4 loss."""
      del preds  # Unused.
      time = utils.bcast_right(time, targets['x0'].ndim)
      alpha = schedule.alpha(time)
      alpha_der = utils.egrad(schedule.alpha)(time)
      alpha = utils.flatten_non_batch_dims(alpha)
      alpha_der = utils.flatten_non_batch_dims(alpha_der)
      weight = -1.0 * alpha_der / jnp.clip(1.0 - alpha, a_min=1e-12)
      return weight

    return compute_discrete_diffusion_loss(
        preds=preds,
        targets=targets,
        time=time,
        schedule=self.schedule,
        use_mask=self.use_mask,
        weight_fn=_weight_fn,
    )
