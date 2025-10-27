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

"""Architecture modules."""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.architecture.arch_typing import ConditioningMechanism
from hackable_diffusion.lib.architecture.arch_typing import DownsampleType
from hackable_diffusion.lib.architecture.arch_typing import EmbeddingMergeMethod
from hackable_diffusion.lib.architecture.arch_typing import NormalizationType
from hackable_diffusion.lib.architecture.arch_typing import RoPEPositionType
from hackable_diffusion.lib.architecture.arch_typing import SkipConnectionMethod
from hackable_diffusion.lib.architecture.arch_typing import UpsampleType
from hackable_diffusion.lib.architecture.attention import MultiHeadAttention
from hackable_diffusion.lib.architecture.conditioning_encoder import BaseConditioningEncoder
from hackable_diffusion.lib.architecture.conditioning_encoder import BaseEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import BaseTimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import ConditioningEncoder
from hackable_diffusion.lib.architecture.conditioning_encoder import IdentityTimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import LabelEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import LinearEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import MLPEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import NestedTimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import SinusoidalTimeEmbedder
from hackable_diffusion.lib.architecture.conditioning_encoder import ZeroTimeEmbedder
from hackable_diffusion.lib.architecture.discrete import BaseTokenEmbedder
from hackable_diffusion.lib.architecture.discrete import BaseTokenProjector
from hackable_diffusion.lib.architecture.discrete import ConditionalDiscreteBackbone
from hackable_diffusion.lib.architecture.discrete import DenseTokenProjector
from hackable_diffusion.lib.architecture.discrete import TokenEmbedder
from hackable_diffusion.lib.architecture.mlp import ConditionalMLP
from hackable_diffusion.lib.architecture.mlp_blocks import MLP
from hackable_diffusion.lib.architecture.sequence_embedders import RandomFourierSequenceEmbedding
from hackable_diffusion.lib.architecture.sequence_embedders import RoPESequenceEmbedding
from hackable_diffusion.lib.architecture.sequence_embedders import SinusoidalSequenceEmbedding
from hackable_diffusion.lib.architecture.unet import Unet
from hackable_diffusion.lib.architecture.unet_blocks import AttentionResidualBlock
from hackable_diffusion.lib.architecture.unet_blocks import ConvResidualBlock
from hackable_diffusion.lib.architecture.unet_blocks import InputConvBlock
from hackable_diffusion.lib.architecture.unet_blocks import OutputConvBlock
# pylint: enable=g-importing-member
