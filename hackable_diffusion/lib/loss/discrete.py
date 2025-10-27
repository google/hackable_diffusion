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
# MARK: Discrete Losses
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiffusionCrossEntropyLoss(base.DiffusionLoss):
  """Cross entropy loss for discrete diffusion.

  In the case where use_mask is True, we use the mask to ignore already revealed
  tokens. In that case, it corresponds to the MD4 Loss, see
  https://arxiv.org/abs/2406.04329 (Eq 5).

    w(t) * α(t)' / (1 - α(t)) * xentropy(targets['x0] | preds['logits']). (1)

  Attributes:
    schedule: The schedule to use for the loss.
    use_mask: Whether to use the mask or not.
    weight_fn: The weight function w(t) to use for the loss, see (1).
  """

  schedule: schedules.DiscreteSchedule
  use_mask: bool = False
  weight_fn: base.WeightFn | None = None

  @typechecked
  def __call__(
      self, preds: TargetInfo, targets: TargetInfo, time: TimeArray
  ) -> LossOutput:
    # The last dimension of preds and targets is a vocabulary dimension.
    time = utils.bcast_right(time, targets['x0'].ndim)
    alpha = self.schedule.alpha(time)
    alpha_der = utils.egrad(self.schedule.alpha)(time)
    alpha = utils.flatten_non_batch_dims(alpha)
    alpha_der = utils.flatten_non_batch_dims(alpha_der)

    bsz = time.shape[0]
    assert alpha.shape == (bsz, 1)
    assert alpha_der.shape == (bsz, 1)

    labels = jnp.squeeze(targets['x0'], axis=-1)
    # Remove trailing dimension of the x0.
    if self.use_mask:
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

    if self.weight_fn is not None:
      weight = self.weight_fn(schedule=self.schedule, time=time)
      weight = utils.flatten_non_batch_dims(weight)
      assert weight.shape == (bsz, 1)
      weighted_loss = weight * alpha_der / (1.0 - alpha) * neg_xentropy
    else:
      weighted_loss = alpha_der / (1.0 - alpha) * neg_xentropy
    weighted_loss = jnp.squeeze(weighted_loss, axis=-1)

    assert weighted_loss.shape == (bsz,)
    return weighted_loss
