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

"""Losses for discrete diffusion."""

import dataclasses
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.training import base
import jax.numpy as jnp
import kauldron.ktyping as kt
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


@kt.typechecked
def _compute_discrete_diffusion_loss(
    preds: TargetInfo,
    targets: TargetInfo,
    time: TimeArray,  # pyrefly: ignore[not-a-type]
    *,
    use_mask: bool = False,
    mask_key: str = 'is_corrupted',
    normalize_by_mask: bool = True,
) -> LossOutput:  # pyrefly: ignore[not-a-type]
  """Compute the discrete diffusion loss."""

  # The last dimension of preds and targets is a vocabulary dimension.
  time = jax_helpers.bcast_right(time, targets['x0'].ndim)

  bsz = time.shape[0]

  labels = jnp.squeeze(targets['x0'], axis=-1)
  # Remove trailing dimension of the x0.
  if use_mask:
    if mask_key not in targets:
      raise ValueError(
          f'Mask key {mask_key} not found in targets: {targets.keys()=}'
      )
    mask = jnp.squeeze(targets[mask_key], axis=-1)
    mask = mask.astype(jnp.bool_)
    # Remove trailing dimension of the mask.
    # Mask is True if xt is corrupted and False otherwise.
  else:
    mask = jnp.ones_like(labels, dtype=jnp.bool_)

  neg_xentropy = -optax.softmax_cross_entropy_with_integer_labels(
      logits=preds['logits'],
      labels=labels,
      where=mask,
  )

  if neg_xentropy.shape != labels.shape:
    raise ValueError(
        f'neg_xentropy shape {neg_xentropy.shape} does not match labels shape'
        f' {labels.shape}'
    )

  # Sum and normalize.
  reduce_axes = tuple(range(1, neg_xentropy.ndim))
  if normalize_by_mask:
    denominator = jnp.sum(mask, axis=reduce_axes, keepdims=True)
  else:
    denominator = jnp.sum(
        jnp.ones_like(neg_xentropy), axis=reduce_axes, keepdims=True
    )
  neg_xentropy = jnp.sum(neg_xentropy, axis=reduce_axes, keepdims=True)
  neg_xentropy = neg_xentropy / jnp.clip(denominator, min=1e-8)
  neg_xentropy = jax_helpers.flatten_non_batch_dims(neg_xentropy)

  if neg_xentropy.shape != (bsz, 1):
    raise ValueError(
        f'neg_xentropy should have shape ({bsz}, 1), got {neg_xentropy.shape}'
    )

  weighted_loss = -1.0 * neg_xentropy

  weighted_loss = jnp.squeeze(weighted_loss, axis=-1)

  if weighted_loss.shape != (bsz,):
    raise ValueError(
        f'weighted_loss should have shape ({bsz},), got {weighted_loss.shape}'
    )
  return weighted_loss


################################################################################
# MARK: Specific Loss functions
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoWeightDiscreteLoss(base.DiffusionLoss):
  """Discrete loss without weight."""

  use_mask: bool = False
  mask_key: str = 'is_corrupted'
  normalize_by_mask: bool = True

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> LossOutput:  # pyrefly: ignore[not-a-type]

    return _compute_discrete_diffusion_loss(
        preds=preds,
        targets=targets,
        time=time,
        use_mask=self.use_mask,
        mask_key=self.mask_key,
        normalize_by_mask=self.normalize_by_mask,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MD4Loss(base.DiffusionLoss):
  """MD4 loss as in https://arxiv.org/abs/2406.04329, Eq 5."""

  schedule: DiscreteSchedule
  use_mask: bool = False
  mask_key: str = 'is_corrupted'
  normalize_by_mask: bool = True

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> LossOutput:  # pyrefly: ignore[not-a-type]
    time_b = jax_helpers.bcast_right(time, targets['x0'].ndim)
    alpha = self.schedule.alpha(time_b)
    alpha_der = jax_helpers.egrad(self.schedule.alpha)(time_b)
    alpha = jax_helpers.flatten_non_batch_dims(alpha)
    alpha_der = jax_helpers.flatten_non_batch_dims(alpha_der)
    weight = -1.0 * alpha_der / jnp.clip(1.0 - alpha, min=1e-12)
    weight = jnp.squeeze(weight, axis=-1)

    loss = _compute_discrete_diffusion_loss(
        preds=preds,
        targets=targets,
        time=time,
        use_mask=self.use_mask,
        mask_key=self.mask_key,
        normalize_by_mask=self.normalize_by_mask,
    )
    return loss * weight
