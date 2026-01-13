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

"""Diffusion Loss functions for Gaussian corruption processes.

For the different prediction and target types, we refer to
hackable_diffusion/lib/corruption/gaussian.py for an in-depth explanation.
"""

import dataclasses
from typing import Literal

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.loss import base
import immutabledict
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

LossOutput = hd_typing.LossOutput
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray
GaussianPredictionType = Literal["x0", "epsilon", "score", "velocity", "v"]

GaussianSchedule = schedules.GaussianSchedule

################################################################################
# MARK: General Loss function
################################################################################


@typechecked
def compute_continuous_diffusion_loss(
    preds: TargetInfo,
    targets: TargetInfo,
    time: TimeArray,
    *,
    schedule: GaussianSchedule | None = None,
    loss_type: GaussianPredictionType | None = None,
    prediction_type: GaussianPredictionType | None = None,
    convert_to_logsnr_schedule: bool = True,
    weight_fn: base.WeightFn | None = None,
) -> LossOutput:
  """Compute the continuous diffusion loss.

  Compute the loss used in the continuous diffusion framework. More precisely,
  assuming that `prediction_type` is accessible from `preds` and `targets`,
  compute the loss defined as:

  Loss = (preds[prediction_type] - targets[prediction_type]) ** 2 * weight

  The weight terms is the product of the following three terms:
  1. A conversion term that transforms the loss to a new prediction type.
  2. A schedule-dependent term that converts the loss to logsnr scale.
  3. An additional weight term that takes the time as argument.
  All three terms are optional and can be left out.

  The first conversion term is only applied if `loss_type` is not None and
  `prediction_type` is not `loss_type`. It can be interpreted as computing

  Loss = (preds[loss_type] - targets[loss_type]) ** 2 * weight

  with the weight term being now only the product of 2) and 3). The term 2) is
  only applied if `convert_to_logsnr_schedule` is True. The term 3) is only
  applied if `weight_fn` is not None.

  We refer to SiD2Loss for a specific implementation of the loss function with
  a Sigmoid weight function.

  Args:
    preds: Prediction dict from the model. Contains one or more of the
      prediction types (x0, epsilon, score, velocity, v).
    targets: Target dict containing the same keys as preds.
    time: Time array used for noise computation.
    schedule: The GaussianSchedule to use for the loss. Should be the same as
      used by the model corruption process. Can be None if the loss is not
      schedule dependent, i.e, convert_to_logsnr_schedule is False and weight_fn
      is None. It might be used in the weighting function or in the conversion
      to logSNR parameterization.
    loss_type: The type of loss to compute. Can be one of {x0, epsilon, score,
      velocity, v}. If None, defaults to prediction_type.
    prediction_type: The type of prediction to compute the loss for. If None,
      defaults to the prediction type in preds, but only if it is unambiguous
      (Raises a ValueError if preds contains multiple prediction types).
    convert_to_logsnr_schedule: Whether to multiply the loss by -1/logsnr_der to
      make it independent of the time parametrization of the schedule.
    weight_fn: A an optional additional weight term. Has to be a function that
      takes the time as argument and returns a weight.

  Returns:
    The batched loss, i.e., a tensor of shape [B,] where B is the batch size. To
    get the scalar loss use `jnp.mean(loss)`. The loss is returned before the
    averaging to allow for other operations such as masking of loss values
    afterwards.
  """

  if convert_to_logsnr_schedule or weight_fn:
    if schedule is None:
      raise ValueError(
          "Schedule must be provided if convert_to_logsnr_schedule or weight_fn"
          " is not None."
      )

  # Auto-detect prediction type if not specified.
  if prediction_type is None:
    if len(preds) != 1:
      raise ValueError(
          "Can only auto-detect prediction type if it is the only prediction"
          f" type. But got {preds.keys()=}"
      )
    prediction_type = next(iter(preds.keys()))

  # Compute MSE part of the loss
  pred = preds[prediction_type]
  target = targets[prediction_type]
  l2 = jnp.square(pred - target)

  # Broadcast time to the same shape as pred
  time = utils.bcast_right(time, pred.ndim)

  # Compute the weight terms
  weight = jnp.ones_like(time)

  # Maybe convert between prediction types
  if loss_type is not None and prediction_type != loss_type:
    if schedule is None:
      raise ValueError(
          "Schedule must be provided if loss_type is not None and if the"
          " provided or inferred prediction_type is not the same as the"
          " loss_type."
      )

    conversion_term = CONVERTERS[prediction_type][loss_type](schedule, time)
    weight = weight * conversion_term

  # Maybe multiply by -dlogsnr/dt
  if schedule is not None and convert_to_logsnr_schedule:
    logsnr_der = utils.egrad(schedule.logsnr)(time)
    weight = -weight * logsnr_der

  # Maybe multiply by other weight terms
  if schedule is not None and weight_fn is not None:
    weight = weight * weight_fn(
        schedule=schedule, preds=preds, targets=targets, time=time
    )

  weighted_loss = weight * l2
  weighted_loss = utils.flatten_non_batch_dims(weighted_loss)
  # We use mean as opposed to sum to make the loss dimension-agnostic.
  weighted_loss = jnp.mean(weighted_loss, axis=-1)

  return weighted_loss


################################################################################
# MARK: Specific Loss functions
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoWeightGaussianLoss(base.DiffusionLoss):
  """Loss without weight."""

  prediction_type: GaussianPredictionType | None = None

  @typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    return compute_continuous_diffusion_loss(
        # arrays
        preds=preds,
        targets=targets,
        time=time,
        # fixed arguments
        loss_type=None,
        convert_to_logsnr_schedule=False,
        weight_fn=None,
        schedule=None,
        prediction_type=self.prediction_type,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SiD2Loss(base.DiffusionLoss):
  """Sigmoid loss as in https://arxiv.org/abs/2410.19324, Equation (4)."""

  bias: float = 0.0
  prediction_type: GaussianPredictionType | None = None
  schedule: GaussianSchedule

  @typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    def _weight_fn(
        schedule: GaussianSchedule,
        preds: TargetInfo,
        targets: TargetInfo,
        time: TimeArray,
    ) -> TimeArray:
      """Weight function for the sigmoid loss.

      The weight function is defined for the `x0` prediction type, see
      https://arxiv.org/abs/2410.19324, Equation (4), i.e., the weight is given
      by

        w_t = sigmoid(logsnr(t) - bias) * exp(bias)

      Args:
        schedule: The GaussianSchedule to use for the loss.
        preds: Unused.
        targets: Unused.
        time: The time array to use for the loss.

      Returns:
        The weight function.
      """
      del preds, targets  # Unused
      return jax.nn.sigmoid(schedule.logsnr(time) - self.bias) * jnp.exp(
          self.bias
      )

    return compute_continuous_diffusion_loss(
        # arrays
        preds=preds,
        targets=targets,
        time=time,
        # fixed arguments
        loss_type="x0",
        convert_to_logsnr_schedule=True,
        weight_fn=_weight_fn,
        # forward arguments
        schedule=self.schedule,
        prediction_type=self.prediction_type,
    )


################################################################################
# MARK: Loss-Type Conv
################################################################################


################################################################################
# MARK: convert from x0
################################################################################


def x0_to_epsilon_scaling(schedule, time: TimeArray) -> TimeArray:
  sigma = schedule.sigma(time)
  alpha = schedule.alpha(time)
  return jnp.square(alpha / sigma)


def x0_to_score_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  return jnp.square(alpha / jnp.square(sigma))


def x0_to_velocity_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  alpha_der = utils.egrad(schedule.alpha)(time)
  sigma_der = utils.egrad(schedule.sigma)(time)
  return jnp.square(alpha_der - alpha * sigma_der / sigma)


def x0_to_v_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  return jnp.square(jnp.square(alpha) / sigma + sigma)


################################################################################
# MARK: convert from epsilon
################################################################################


def epsilon_to_x0_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / x0_to_epsilon_scaling(schedule, time)


def epsilon_to_score_scaling(schedule, time: TimeArray) -> TimeArray:
  sigma = schedule.sigma(time)
  return 1.0 / jnp.square(sigma)


def epsilon_to_velocity_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  alpha_der = utils.egrad(schedule.alpha)(time)
  sigma_der = utils.egrad(schedule.sigma)(time)
  return jnp.square(sigma * alpha_der / alpha - sigma_der)


def epsilon_to_v_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  return jnp.square(alpha + jnp.square(sigma) / alpha)


################################################################################
# MARK: convert from score
################################################################################


def score_to_x0_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / x0_to_score_scaling(schedule, time)


def score_to_epsilon_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / epsilon_to_score_scaling(schedule, time)


def score_to_velocity_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  alpha_der = utils.egrad(schedule.alpha)(time)
  sigma_der = utils.egrad(schedule.sigma)(time)
  return jnp.square(jnp.square(sigma) * alpha_der / alpha - sigma * sigma_der)


def score_to_v_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  return jnp.square(sigma * alpha + jnp.power(sigma, 3) / alpha)


################################################################################
# MARK: convert from velocity
################################################################################


def velocity_to_x0_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / x0_to_velocity_scaling(schedule, time)


def velocity_to_epsilon_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / epsilon_to_velocity_scaling(schedule, time)


def velocity_to_score_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / score_to_velocity_scaling(schedule, time)


def velocity_to_v_scaling(schedule, time: TimeArray) -> TimeArray:
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  alpha_der = utils.egrad(schedule.alpha)(time)
  sigma_der = utils.egrad(schedule.sigma)(time)
  return jnp.square(
      (jnp.square(alpha) + jnp.square(sigma))
      / (sigma * alpha_der - alpha * sigma_der)
  )


#################################################################################
# MARK: convert from v
################################################################################


def v_to_x0_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / x0_to_v_scaling(schedule, time=time)


def v_to_epsilon_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / epsilon_to_v_scaling(schedule, time=time)


def v_to_score_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / score_to_v_scaling(schedule, time=time)


def v_to_velocity_scaling(schedule, time: TimeArray) -> TimeArray:
  return 1.0 / velocity_to_v_scaling(schedule, time=time)


def _identity(schedule, time: TimeArray) -> TimeArray:
  del schedule  # Unused
  return jnp.ones_like(time)


def _not_implemented(schedule, time: TimeArray) -> TimeArray:
  del schedule, time  # Unused
  raise NotImplementedError("Not implemented")


CONVERTERS = immutabledict.immutabledict({
    "x0": {
        "x0": _identity,
        "epsilon": x0_to_epsilon_scaling,
        "score": x0_to_score_scaling,
        "velocity": x0_to_velocity_scaling,
        "v": x0_to_v_scaling,
    },
    "epsilon": {
        "x0": epsilon_to_x0_scaling,
        "epsilon": _identity,
        "score": epsilon_to_score_scaling,
        "velocity": epsilon_to_velocity_scaling,
        "v": epsilon_to_v_scaling,
    },
    "score": {
        "x0": score_to_x0_scaling,
        "epsilon": score_to_epsilon_scaling,
        "score": _identity,
        "velocity": score_to_velocity_scaling,
        "v": score_to_v_scaling,
    },
    "velocity": {
        "x0": velocity_to_x0_scaling,
        "epsilon": velocity_to_epsilon_scaling,
        "score": velocity_to_score_scaling,
        "velocity": _identity,
        "v": velocity_to_v_scaling,
    },
    "v": {
        "x0": v_to_x0_scaling,
        "epsilon": v_to_epsilon_scaling,
        "score": v_to_score_scaling,
        "velocity": v_to_velocity_scaling,
        "v": _identity,
    },
})
