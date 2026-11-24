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

from typing import Any, Mapping, TypeAlias
import kauldron.ktyping as kt

# pylint: disable=g-multiple-import,g-importing-member, unused-import
from kauldron.ktyping import (
    Array,
    BFloat16,
    Bool,
    Complex,
    Complex64,
    Float,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    Num,
    PRNGKey,
    PyTree,
    Scalar,
    SInt,
    UInt,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

# pylint: enable=g-multiple-import,g-importing-member, unused-import

typechecked = kt.typechecked
check_type = kt.check_type


# MARK: Data Structure

# We define batched structures and corresponding PyTree structures for
# all important modalities. The first dimension of any batched structure is
# assumed to be the batch dimension.

# Array described the batched data.
DataArray: TypeAlias = Array['batch *#data_shape']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

Scalar: TypeAlias = Array['batch']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

# Array of the shape and structure of the time parameter.
# '*#data_shape' means broadcastable to the shape of the data.
# So (B, 1, 1, 1) would be ok assuming overall shape is (B, H, W, C).
# TODO(b/493016456): TimeArray should use `*#data_shape` like DataArray, but
# time sometimes has shape (B,) instead of (B, 1, 1, 1), so we use non-binding.
TimeArray: TypeAlias = Array['#batch *_data_shape']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

# Corresponding schedule.
ScheduleKey: TypeAlias = str  # e.g. 'time', 'alpha', 'sigma', 'logsnr', etc.

# A dictionary containing the different training targets. Same structure as
# DataArray for every different target (e.g. x0, epsilon, score, velocity,
# v, mask, ...).
# NOTE: The # in data_shape is there because in the discrete case the targets
# are usually labels (x0 : Int["batch 1"]) while the predictions are
# logits (x0 : Float["batch K"]).
TargetKey: TypeAlias = str  # e.g. 'x0', 'epsilon', 'score', 'velocity', ...
TargetInfo: TypeAlias = Mapping[TargetKey, Array['batch *_data_shape']]  # pyrefly: ignore[not-a-type, unknown-name]

# Conditioning structures.
ConditioningKey: TypeAlias = str  # e.g. 'label', 'text', 'image', ...
Conditioning: TypeAlias = Mapping[ConditioningKey, Any]

ConditioningEmbeddingsKey: TypeAlias = str  # e.g. 'adaptive_norm', ...
ConditioningEmbeddings: TypeAlias = dict[ConditioningEmbeddingsKey, Any]


# Shape related structures.
Shape: TypeAlias = tuple[int, ...]
ConditioningShape: TypeAlias = dict[ConditioningKey, Shape]

# Type related structures.
DType: TypeAlias = kt.DType


# Loss related structures.
LossOutput: TypeAlias = Float['batch']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]


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
DataTree: TypeAlias = PyTree[Array['batch *_data_shape'], '$T']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

ScalarTree: TypeAlias = PyTree[Array['batch'], '$T']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

# Corresponding PyTree for the time array.
TimeTree: TypeAlias = PyTree[Array['_batch *_data_shape'], '$T']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

ScheduleInfoTree: TypeAlias = PyTree[  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]
    dict[ScheduleKey, Array['batch *_data_shape']], '$T'  # pyrefly: ignore[not-a-type, unknown-name]
]

TargetInfoTree: TypeAlias = PyTree[Array['batch *_data_shape']]  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]

ShapeTree: TypeAlias = PyTree[Shape]  # pyrefly: ignore[invalid-type-alias]

DTypeTree: TypeAlias = PyTree[DType, '$T']  # pyrefly: ignore[invalid-type-alias, unknown-name]

LossOutputTree: TypeAlias = PyTree[LossOutput, '$T']  # pyrefly: ignore[invalid-type-alias, not-a-type, unknown-name]



