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

"""Common typing definitions."""

from collections.abc import Mapping  # pylint: disable=g-multiple-import,g-importing-member
# pylint: disable=g-multiple-import,g-importing-member, unused-import
from hackable_diffusion.lib.array_annotations import (
    check_type,
    BFloat16,
    Bool,
    Complex,
    Complex64,
    Float,
    Float32,
    Float64,
    Int,
    Int32,
    Int64,
    Int8,
    Int16,
    Array,
    Num,
    SInt,
    UInt,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    typechecked,
    PRNGKey,
    Scalar,
)
# pylint: enable=g-multiple-import,g-importing-member, unused-import
from jaxtyping import PyTree, DTypeLike  # pylint: disable=g-multiple-import,g-importing-member


# MARK: Data Structure

# We define batched structures and corresponding PyTree structures for
# all important modalities. The first dimension of any batched structure is
# assumed to be the batch dimension.

# Array described the batched data.
DataArray = Array['batch *#data_shape']
# PyTree of the shape and structure of the input data.
# Note: the ? prefix means that the data_shape can be different for different
# leaves of the PyTree.
DataTree = PyTree[Array['batch *?#data_shape'], 'TD']


# Array of the shape and structure of the time parameter.
# '*#data_shape' means broadcastable to the shape of the data.
# So (B, 1, 1, 1) would be ok assuming overall shape is (B, H, W, C).
TimeArray = Array['#batch *#data_shape']
# Corresponding PyTree for the time array.
TimeTree = PyTree[Array['#batch *?#data_shape'], 'TD']
# Corresponding schedule tree.
ScheduleInfoTree = PyTree[dict[str, Array['batch *?#data_shape']], 'TD']

# A dictionary containing the different training targets. Same structure as
# DataArray for every different target (e.g. x0, epsilon, score, velocity,
# v, mask, ...).
# NOTE: The # in data_shape is there because in the discrete case the targets
# are usually labels (x0 : Int["batch 1"]) while the predictions are
# logits (x0 : Float["batch K"]).
TargetInfo = dict[str, Array['batch *#data_shape']]
TargetInfoTree = PyTree[dict[str, Array['batch *?#data_shape']], 'TD']

# Conditioning structures.
Conditioning = Mapping[str, Array['batch *cond_shape']]

# Shape related structures.
Shape = tuple[int, ...]
ShapeTree = PyTree[Shape]
ConditioningShape = dict[str, Shape]

# Type related structures.
DType = DTypeLike
DTypeTree = PyTree[DType, 'TD']


# Loss related structures.
LossOutput = Float['batch']
LossOutputTree = PyTree[LossOutput, 'TD']
