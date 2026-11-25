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

"""Projection."""

import dataclasses
from typing import Protocol, TypeAlias, cast
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import jax_helpers
import jax
import jax.numpy as jnp


################################################################################
# MARK: Type Aliases
################################################################################

Batch: TypeAlias = hd_api.Batch
Conditioning: TypeAlias = hd_api.Conditioning
typechecked = jax_helpers.typechecked


################################################################################
# MARK: Protocols
################################################################################


class ProjectionFn[Operand](Protocol):
  """Projection function protocol."""

  def __call__(
      self,
      xt: hd_api.Data[Operand],
      conditioning: Conditioning | None,
      time: hd_api.Time[Operand],
      outputs: hd_api.TargetInfo[Operand],
  ) -> hd_api.TargetInfo[Operand]:
    """Projection function protocol."""
    ...


################################################################################
# MARK: Helper Functions
################################################################################


@typechecked
def _to_x0(
    preds: hd_api.TargetInfo[Batch],
    xt: hd_api.Data[Batch],
    time: hd_api.Time[Batch],
    process: hd_api.CorruptionProcess[Batch],
) -> tuple[str, hd_api.TargetInfo[Batch]]:
  """Converts the preds to x0 prediction.

  Convert the preds to x0 prediction by applying the corruption process
  `convert_predictions` method. It also returns the initial prediction type.

  Args:
    preds: The prediction.
    xt: The noisy data.
    time: The time of the diffusion process.
    process: The corruption process to use for conversion.

  Returns:
    The prediction type and the x0 prediction.
  """
  if len(preds) != 1:
    raise ValueError(
        f"Projection only supports single prediction type, got {preds.keys()=}"
    )
  prediction_type = next(iter(preds.keys()))
  all_preds = process.convert_predictions(
      prediction=preds,
      xt=xt,
      time=time,
  )
  return prediction_type, {"x0": all_preds["x0"]}


@typechecked
def _from_x0(
    preds: hd_api.TargetInfo[Batch],
    xt: hd_api.Data[Batch],
    time: hd_api.Time[Batch],
    process: hd_api.CorruptionProcess[Batch],
    prediction_type: str,
) -> hd_api.TargetInfo[Batch]:
  """Converts the x0 prediction to prediction type.

  This is the inverse of `_to_x0`. Given a prediction type and the x0
  prediction, it applies the corruption process `convert_predictions` method.

  Args:
    preds: The prediction (must be x0-prediction).
    xt: The noisy data.
    time: The time of the diffusion process.
    process: The corruption process to use for conversion.
    prediction_type: The prediction type to convert to.

  Returns:
    The prediction type and the x0 prediction.
  """
  if tuple(preds.keys()) != ("x0",):
    raise ValueError(
        f"Projection only supports x0 prediction type, got {preds.keys()=}"
    )
  all_preds = process.convert_predictions(
      prediction=preds,
      xt=xt,
      time=time,
  )
  return cast(
      hd_api.TargetInfo[Batch], {prediction_type: all_preds[prediction_type]}
  )


################################################################################
# MARK: Projection Functions
################################################################################


class IdentityProjectionFn(ProjectionFn[Batch]):
  """Identity projection function."""

  @typechecked
  def __call__(
      self,
      xt: hd_api.Data[Batch],
      conditioning: Conditioning | None,
      time: hd_api.Time[Batch],
      outputs: hd_api.TargetInfo[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    """Identity projection function."""
    del xt, conditioning, time  # unused
    return outputs


@dataclasses.dataclass(kw_only=True, frozen=True)
class StaticThresholdProjectionFn(ProjectionFn[Batch]):
  """Static threshold projection function.

  Attributes:
    min_value: The minimum value to clip the x0 prediction to.
    max_value: The maximum value to clip the x0 prediction to.
    process: The corruption process to use for conversion.
  """

  min_value: float = -1.0
  max_value: float = 1.0
  process: hd_api.CorruptionProcess[Batch]

  def __post_init__(self):
    if self.min_value > self.max_value:
      raise ValueError(
          "Min value must be less than or equal to max value, got"
          f" {self.min_value=} and {self.max_value=}"
      )

  @typechecked
  def __call__(
      self,
      xt: hd_api.Data[Batch],
      conditioning: Conditioning | None,
      time: hd_api.Time[Batch],
      outputs: hd_api.TargetInfo[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    """Static threshold projection function."""
    del conditioning  # unused
    prediction_type, x0_preds = _to_x0(outputs, xt, time, self.process)
    x0_preds = jax.tree.map(
        lambda x: jnp.clip(x, self.min_value, self.max_value), x0_preds
    )
    return _from_x0(x0_preds, xt, time, self.process, prediction_type)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DynamicThresholdProjectionFn(ProjectionFn[Batch]):
  """Dynamic threshold projection function.

  The implementation is based on https://arxiv.org/abs/2205.11487.

  Attributes:
    percentile: The percentile to use for the dynamic threshold. Must be between
      0.0 and 100.0. Default is 95.0.
    process: The corruption process to use for conversion.
  """

  percentile: float = 95.0
  process: hd_api.CorruptionProcess[Batch]

  def __post_init__(self):
    if self.percentile < 0.0 or self.percentile > 100.0:
      raise ValueError(
          f"Percentile must be between 0.0 and 100.0, got {self.percentile}"
      )

  def _dynamic_threshold(self, x0: hd_api.Data[Batch]) -> hd_api.Data[Batch]:
    """Dynamic threshold projection function."""
    axes = tuple(range(1, x0.ndim))
    s = jnp.percentile(jnp.abs(x0), self.percentile, axis=axes, keepdims=True)
    s = jnp.maximum(s, 1.0)
    x0 = jnp.clip(x0, -s, s) / s
    return x0

  @typechecked
  def __call__(
      self,
      xt: hd_api.Data[Batch],
      conditioning: Conditioning | None,
      time: hd_api.Time[Batch],
      outputs: hd_api.TargetInfo[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    """Dynamic threshold projection function."""
    del conditioning  # unused

    prediction_type, x0_preds = _to_x0(outputs, xt, time, self.process)
    x0_preds = jax.tree.map(self._dynamic_threshold, x0_preds)
    return _from_x0(x0_preds, xt, time, self.process, prediction_type)
