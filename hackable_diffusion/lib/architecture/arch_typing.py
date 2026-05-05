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

"""Commonly used types and protocols for the architecture.

Please refer to individual modules for more detailed documentation and
definitions of the components.
"""

import enum
from typing import Any
from typing import Callable, Protocol
from hackable_diffusion.lib import hd_typing
import jax


################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float

DataTree = hd_typing.DataTree

################################################################################
# MARK: Constants
################################################################################

INVALID_INT = -1

################################################################################
# MARK: Enums
################################################################################


class EmbeddingMergeMethod(enum.StrEnum):
  """Methods for merging embeddings in the conditioning encoder."""

  SUM = "sum"
  CONCAT = "concat"


class RoPEPositionType(enum.StrEnum):
  """Rotary Position Embedding (RoPE) types."""

  SQUARE = "square"
  LINEAR = "linear"


class NormalizationType(enum.StrEnum):
  RMS_NORM = "rms_norm"
  GROUP_NORM = "group_norm"
  LAYER_NORM = "layer_norm"


class DownsampleType(enum.StrEnum):
  """Image downsampling methods."""

  MAX_POOL = "max_pool"
  AVG_POOL = "avg_pool"


class UpsampleType(enum.StrEnum):
  """Image upsampling methods."""

  NEAREST = "nearest"
  BILINEAR = "bilinear"


class SkipConnectionMethod(enum.StrEnum):
  """Methods for adding skip connections."""

  UNNORMALIZED_ADD = "unnormalized_add"
  NORMALIZED_ADD = "normalized_add"


################################################################################
# MARK: Conditioning Mechanism
################################################################################


class ConditioningMechanism(enum.StrEnum):
  """Types of conditioning mechanisms."""

  ADAPTIVE_NORM = "adaptive_norm"
  CROSS_ATTENTION = "cross_attention"
  CONCATENATE = "concatenate"
  SUM = "sum"
  SELF_CONDITIONING = "self_conditioning"
  CUSTOM = "custom"


# Conditioning embeddings corresponds to a dictionary with keys corresponding to
# the specification of a conditioning mechanism. We use `ConditioningMechanism`
# as a reference for the most common conditioning mechanisms, but the type
# structure is more flexible to allow for more general conditioning mechanisms.
ConditioningEmbeddings = dict[str, Any]


################################################################################
# MARK: Types and protocols
################################################################################

ActivationFn = Callable[[jax.Array], jax.Array]


class ConditionalBackbone(Protocol):
  """Protocol for a conditional backbone."""

  def __call__(
      self,
      x: DataTree,
      conditioning_embeddings: ConditioningEmbeddings,
      is_training: bool,
  ) -> DataTree:
    ...


class SkipConnectionFn(Protocol):
  """Skip connection function."""

  def __call__(
      self,
      x: Float["batch height width channels"],
      skip: Float["batch height width channels"],
  ) -> Float["batch height width channels"]:
    ...


class DownsampleFn(Protocol):
  """Downsample function."""

  def __call__(
      self,
      x: Float["batch height width channels"],
  ) -> Float["batch height//2 width//2 channels"]:
    ...


class UpsampleFn(Protocol):
  """Upsample function."""

  def __call__(
      self,
      x: Float["batch height width channels"],
  ) -> Float["batch 2*height 2*width channels"]:
    ...
