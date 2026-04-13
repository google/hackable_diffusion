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

from collections.abc import Mapping  # pylint: disable=g-multiple-import,g-importing-member
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
DataArray = Array['batch *#data_shape']
# PyTree of the shape and structure of the input data.
# Note: the _ prefix means that the data_shape can be different for different
# leaves of the PyTree (non-binding dim). This replaces jaxtyping's `?#`
# combined prefix which ktyping does not support.

DataTree = PyTree[Array['batch *_data_shape'], '$T']


# Array of the shape and structure of the time parameter.
# '*#data_shape' means broadcastable to the shape of the data.
# So (B, 1, 1, 1) would be ok assuming overall shape is (B, H, W, C).
# TODO(b/493016456): TimeArray should use `*#data_shape` like DataArray, but
# time sometimes has shape (B,) instead of (B, 1, 1, 1), so we use non-binding.
TimeArray = Array['#batch *_data_shape']
# Corresponding PyTree for the time array.
TimeTree = PyTree[Array['_batch *_data_shape'], '$T']

# Corresponding schedule tree.
ScheduleKey = str  # e.g. 'time', 'alpha', 'sigma', 'logsnr', etc.
ScheduleInfoTree = PyTree[dict[ScheduleKey, Array['batch *_data_shape']], '$T']

# A dictionary containing the different training targets. Same structure as
# DataArray for every different target (e.g. x0, epsilon, score, velocity,
# v, mask, ...).
# NOTE: The # in data_shape is there because in the discrete case the targets
# are usually labels (x0 : Int["batch 1"]) while the predictions are
# logits (x0 : Float["batch K"]).
TargetKey = str  # e.g. 'x0', 'epsilon', 'score', 'velocity', 'v', 'mask', ...
TargetInfo = dict[TargetKey, Array['batch *_data_shape']]
TargetInfoTree = PyTree[Array['batch *_data_shape']]

# Conditioning structures.
ConditioningKey = str  # e.g. 'label', 'text', 'image', ...
Conditioning = Mapping[ConditioningKey, Array['batch *cond_shape']]

# Shape related structures.
Shape = tuple[int, ...]
ShapeTree = PyTree[Shape]
ConditioningShape = dict[ConditioningKey, Shape]

# Type related structures.
DType = kt.DType
DTypeTree = PyTree[DType, '$T']


# Loss related structures.
LossOutput = Float['batch']
LossOutputTree = PyTree[LossOutput, '$T']
