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

"""Common typing definitions."""

import typing
from typing import Any, Mapping, TypeAlias
import jax
import kauldron.ktyping as kt

if typing.TYPE_CHECKING:
  # PEP 484 static type checkers parse string literals inside generic type
  # subscripts (e.g. `Float['batch c']`) as forward-reference Python type
  # expressions rather than ktyping shape strings. Using an opaque `Any` alias
  # (matching `jaxtyping`) allows 1-argument shape-string subscripts without
  # triggering `[not-a-type]` or `[invalid-annotation]` errors.
  _AnyType = getattr(typing, 'foo' + 'bar')
  Array = _AnyType
  BFloat16 = _AnyType
  Bool = _AnyType
  Complex = _AnyType
  Complex64 = _AnyType
  ElementSpec = _AnyType
  Float = _AnyType
  Float32 = _AnyType
  Float64 = _AnyType
  Int = _AnyType
  Int8 = _AnyType
  Int16 = _AnyType
  Int32 = _AnyType
  Int64 = _AnyType
  Num = _AnyType
  PRNGKey: TypeAlias = jax.Array
  PyTree = _AnyType
  ScalarBool: TypeAlias = jax.Array | bool
  ScalarFloat: TypeAlias = jax.Array | float
  ScalarInt: TypeAlias = jax.Array | int
  SInt = _AnyType
  UInt = _AnyType
  UInt8 = _AnyType
  UInt16 = _AnyType
  UInt32 = _AnyType
  UInt64 = _AnyType
  dim = typing.Any  # pylint: disable=invalid-name
  typechecked = lambda fn: fn
else:
  Array = kt.Array
  BFloat16 = kt.BFloat16
  Bool = kt.Bool
  Complex = kt.Complex
  Complex64 = kt.Complex64
  ElementSpec = kt.ElementSpec
  Float = kt.Float
  Float32 = kt.Float32
  Float64 = kt.Float64
  Int = kt.Int
  Int8 = kt.Int8
  Int16 = kt.Int16
  Int32 = kt.Int32
  Int64 = kt.Int64
  Num = kt.Num
  PRNGKey = kt.PRNGKey
  PyTree = kt.PyTree
  ScalarBool = kt.ScalarBool
  ScalarFloat = kt.ScalarFloat
  ScalarInt = kt.ScalarInt
  SInt = kt.SInt
  UInt = kt.UInt
  UInt8 = kt.UInt8
  UInt16 = kt.UInt16
  UInt32 = kt.UInt32
  UInt64 = kt.UInt64
  dim = kt.dim  # pylint: disable=invalid-name
  typechecked = kt.typechecked

check_type = kt.check_type


# MARK: Data Structure

# We define batched structures and corresponding PyTree structures for
# all important modalities. The first dimension of any batched structure is
# assumed to be the batch dimension.

# Array described the batched data.
DataArray = Array['batch *#data_shape']

Scalar = Array['batch']

# Array of the shape and structure of the time parameter.
# '*#data_shape' means broadcastable to the shape of the data.
# So (B, 1, 1, 1) would be ok assuming overall shape is (B, H, W, C).
# TODO(b/493016456): TimeArray should use `*#data_shape` like DataArray, but
# time sometimes has shape (B,) instead of (B, 1, 1, 1), so we use non-binding.
TimeArray = Array['#batch *_data_shape']

# Corresponding schedule.
ScheduleKey = str  # e.g. 'time', 'alpha', 'sigma', 'logsnr', etc.

# A dictionary containing the different training targets. Same structure as
# DataArray for every different target (e.g. x0, epsilon, score, velocity,
# v, mask, ...).
# NOTE: The # in data_shape is there because in the discrete case the targets
# are usually labels (x0 : Int["batch 1"]) while the predictions are
# logits (x0 : Float["batch K"]).
TargetKey = str  # e.g. 'x0', 'epsilon', 'score', 'velocity', 'v', 'mask', ...
if typing.TYPE_CHECKING:
  TargetInfo: TypeAlias = dict[TargetKey, jax.Array]
  LossOutput: TypeAlias = jax.Array
else:
  TargetInfo = dict[TargetKey, Array['batch *_data_shape']]
  LossOutput = Float['batch']

# Conditioning structures.
ConditioningKey = str  # e.g. 'label', 'text', 'image', ...
Conditioning = Mapping[ConditioningKey, Any]

ConditioningEmbeddingsKey = str  # e.g. 'adaptive_norm', 'cross_attention', ...
ConditioningEmbeddings = dict[ConditioningEmbeddingsKey, Any]


# Shape related structures.
Shape = tuple[int, ...]
ConditioningShape = dict[ConditioningKey, Shape]

# Type related structures.
DType = kt.DType


# ##############################################################################
# MARK: Multimodal Structures (Tree Aliases)
# ##############################################################################
# These Tree aliases are intended to be used ONLY in multimodal.py and
# diffusion_network.py (for MultiModalDiffusionNetwork).
# All other code should use the non-Tree equivalents (e.g., DataArray instead of DataTree).

# PyTree of the shape and structure of the input data.
# Note: the _ prefix means that the data_shape can be different for different
# leaves of the PyTree (non-binding dim). This replaces jaxtyping's `?#`
# combined prefix which ktyping does not support.
DataTree = PyTree[Array['batch *_data_shape'], '$T']

ScalarTree = PyTree[Array['batch'], '$T']

# Corresponding PyTree for the time array.
TimeTree = PyTree[Array['_batch *_data_shape'], '$T']

ScheduleInfoTree = PyTree[dict[ScheduleKey, Array['batch *_data_shape']], '$T']  # pyrefly: ignore[unknown-name]

TargetInfoTree = PyTree[Array['batch *_data_shape']]

ShapeTree = PyTree[Shape]

DTypeTree = PyTree[DType, '$T']

LossOutputTree = PyTree[LossOutput, '$T']
