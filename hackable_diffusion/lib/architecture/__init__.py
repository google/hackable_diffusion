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

"""Architecture modules."""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.architecture.attention import AttentionHeadsSpec
from hackable_diffusion.lib.architecture.attention import MultiHeadAttention
from hackable_diffusion.lib.architecture.conditioning_encoder import ConcatEmbeddings
from hackable_diffusion.lib.architecture.conditioning_encoder import ConditioningEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import ConditioningEncoder
from hackable_diffusion.lib.architecture.conditioning_encoder import FieldSelector
from hackable_diffusion.lib.architecture.conditioning_encoder import IdentityTimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import LabelEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import LinearEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import MergeEmbeddingsFn
from hackable_diffusion.lib.architecture.conditioning_encoder import MLPEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import SinusoidalTimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import StandardConditioningEncoder
from hackable_diffusion.lib.architecture.conditioning_encoder import SumEmbeddings
from hackable_diffusion.lib.architecture.conditioning_encoder import TimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import ZeroTimeEmbedder
from hackable_diffusion.lib.architecture.discrete import ConditionalDiscreteBackbone
from hackable_diffusion.lib.architecture.discrete import DenseProjector
from hackable_diffusion.lib.architecture.discrete import TokenEmbedder
from hackable_diffusion.lib.architecture.discrete import TokenProjector
from hackable_diffusion.lib.architecture.dit import DiT
from hackable_diffusion.lib.architecture.dit_blocks import DePatchify
from hackable_diffusion.lib.architecture.dit_blocks import DiTBlock
from hackable_diffusion.lib.architecture.dit_blocks import DiTBlockAdaLNZero
from hackable_diffusion.lib.architecture.dit_blocks import DiTBlockFlux
from hackable_diffusion.lib.architecture.dit_blocks import DiTBlockSD3
from hackable_diffusion.lib.architecture.dit_blocks import Patchify
from hackable_diffusion.lib.architecture.mlp import ConditionalMLP
from hackable_diffusion.lib.architecture.mlp_blocks import MLP
from hackable_diffusion.lib.architecture.normalization import ConditionalGroupNormStrategy
from hackable_diffusion.lib.architecture.normalization import ConditionalLayerNormStrategy
from hackable_diffusion.lib.architecture.normalization import ConditionalRMSNormStrategy
from hackable_diffusion.lib.architecture.normalization import GroupNormStrategy
from hackable_diffusion.lib.architecture.normalization import LayerNormStrategy
from hackable_diffusion.lib.architecture.normalization import NormStrategy
from hackable_diffusion.lib.architecture.normalization import RMSNormStrategy
from hackable_diffusion.lib.architecture.riemannian import RiemannianConditionalBackbone
from hackable_diffusion.lib.architecture.sequence_embedders import LinearRoPEPositions
from hackable_diffusion.lib.architecture.sequence_embedders import RandomFourierSequenceEmbedding
from hackable_diffusion.lib.architecture.sequence_embedders import RoPEPositionsFn
from hackable_diffusion.lib.architecture.sequence_embedders import RoPESequenceEmbedding
from hackable_diffusion.lib.architecture.sequence_embedders import SinusoidalSequenceEmbedding
from hackable_diffusion.lib.architecture.sequence_embedders import SquareRoPEPositions
from hackable_diffusion.lib.architecture.simplicial import ConditionalSimplicialBackbone
from hackable_diffusion.lib.architecture.simplicial import DenseLogitEmbedder
from hackable_diffusion.lib.architecture.simplicial import LogitEmbedder
from hackable_diffusion.lib.architecture.unet import Unet
from hackable_diffusion.lib.architecture.unet_blocks import AttentionResidualBlock
from hackable_diffusion.lib.architecture.unet_blocks import AvgPoolDownsample
from hackable_diffusion.lib.architecture.unet_blocks import ConvResidualBlock
from hackable_diffusion.lib.architecture.unet_blocks import DownsampleFn
from hackable_diffusion.lib.architecture.unet_blocks import ImageResizeUpsample
from hackable_diffusion.lib.architecture.unet_blocks import InputConvBlock
from hackable_diffusion.lib.architecture.unet_blocks import MaxPoolDownsample
from hackable_diffusion.lib.architecture.unet_blocks import NormalizedAddSkip
from hackable_diffusion.lib.architecture.unet_blocks import OutputConvBlock
from hackable_diffusion.lib.architecture.unet_blocks import SkipConnectionFn
from hackable_diffusion.lib.architecture.unet_blocks import UnnormalizedAddSkip
from hackable_diffusion.lib.architecture.unet_blocks import UpsampleFn
# pylint: enable=g-importing-member
