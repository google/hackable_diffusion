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
from typing import Protocol
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray


################################################################################
# MARK: Protocols
################################################################################


class ProjectionFn(Protocol):
  """Projection function protocol."""

  def __call__(
      self,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      outputs: TargetInfo,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfo:  # pyrefly: ignore[not-a-type]
    """Projection function protocol."""
    ...


################################################################################
# MARK: Helper Functions
################################################################################


@kt.typechecked
def _to_x0(
    preds: TargetInfo,
    xt: DataArray,  # pyrefly: ignore[not-a-type]
    time: TimeArray,  # pyrefly: ignore[not-a-type]
    process: hd_api.CorruptionProcess,
) -> tuple[str, TargetInfo]:
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


@kt.typechecked
def _from_x0(
    preds: TargetInfo,
    xt: DataArray,  # pyrefly: ignore[not-a-type]
    time: TimeArray,  # pyrefly: ignore[not-a-type]
    process: hd_api.CorruptionProcess,
    prediction_type: str,
) -> TargetInfo:
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
  return {prediction_type: all_preds[prediction_type]}


################################################################################
# MARK: Projection Functions
################################################################################


class IdentityProjectionFn(ProjectionFn):
  """Identity projection function."""

  def __call__(
      self,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      outputs: TargetInfo,
  ) -> TargetInfo:
    """Identity projection function."""
    del xt, conditioning, time  # unused
    return outputs


@dataclasses.dataclass(kw_only=True, frozen=True)
class StaticThresholdProjectionFn(ProjectionFn):
  """Static threshold projection function.

  Attributes:
    min_value: The minimum value to clip the x0 prediction to.
    max_value: The maximum value to clip the x0 prediction to.
    process: The corruption process to use for conversion.
  """

  min_value: float = -1.0
  max_value: float = 1.0
  process: hd_api.CorruptionProcess

  def __post_init__(self):
    if self.min_value > self.max_value:
      raise ValueError(
          "Min value must be less than or equal to max value, got"
          f" {self.min_value=} and {self.max_value=}"
      )

  @kt.typechecked
  def __call__(
      self,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      outputs: TargetInfo,
  ) -> TargetInfo:
    """Static threshold projection function."""
    del conditioning  # unused
    prediction_type, x0_preds = _to_x0(outputs, xt, time, self.process)
    x0_preds = jax.tree.map(
        lambda x: jnp.clip(x, self.min_value, self.max_value), x0_preds
    )
    return _from_x0(x0_preds, xt, time, self.process, prediction_type)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DynamicThresholdProjectionFn(ProjectionFn):
  """Dynamic threshold projection function.

  The implementation is based on https://arxiv.org/abs/2205.11487.

  Attributes:
    percentile: The percentile to use for the dynamic threshold. Must be between
      0.0 and 100.0. Default is 95.0.
    process: The corruption process to use for conversion.
  """

  percentile: float = 95.0
  process: hd_api.CorruptionProcess

  def __post_init__(self):
    if self.percentile < 0.0 or self.percentile > 100.0:
      raise ValueError(
          f"Percentile must be between 0.0 and 100.0, got {self.percentile}"
      )

  def _dynamic_threshold(self, x0: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Dynamic threshold projection function."""
    axes = tuple(range(1, x0.ndim))
    s = jnp.percentile(jnp.abs(x0), self.percentile, axis=axes, keepdims=True)
    s = jnp.maximum(s, 1.0)
    x0 = jnp.clip(x0, -s, s) / s
    return x0

  @kt.typechecked
  def __call__(
      self,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      outputs: TargetInfo,
  ) -> TargetInfo:
    """Dynamic threshold projection function."""

    prediction_type, x0_preds = _to_x0(outputs, xt, time, self.process)
    x0_preds = jax.tree.map(self._dynamic_threshold, x0_preds)
    return _from_x0(x0_preds, xt, time, self.process, prediction_type)
