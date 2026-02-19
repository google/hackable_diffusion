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

"""DiT architecture."""

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import dit_blocks
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import sequence_embedders
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp

################################################################################
# MARK: Common types
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

ConditionalBackbone = arch_typing.ConditionalBackbone
ConditioningMechanism = arch_typing.ConditioningMechanism
RoPEPositionType = arch_typing.RoPEPositionType

################################################################################
# MARK: DiT
################################################################################


class DiT(ConditionalBackbone):
  """Diffusion Transformer (DiT) backbone.

  Based on
  https://arxiv.org/abs/2212.09748
  https://github.com/facebookresearch/DiT

  Main architectural differences from official implementation:
  - Use learned positional embeddings instead of 2D sinusoidal embeddings.


  Attributes:
    patch_size: The size of the patches.
    hidden_size: The dimension of the embedding.
    depth: The number of DiT blocks.
    num_heads: The number of attention heads.
    mlp_ratio: The ratio of the hidden dimension in the MLP to the input
      dimension.
    dropout_rate: The dropout rate. Not used since DiT does not use dropout.
    attention_use_rope: Whether to use rotary positional embeddings.
    attention_rope_position_type: The type of rotary positional embeddings.
    normalization_type: The normalization method to use.
    normalization_num_groups: The number of groups for group normalization.
    output_channels: The number of output channels. If None, defaults to the
      number of input channels.
    dtype: The data type of the computation.
  """

  patch_size: int
  hidden_size: int
  depth: int
  num_heads: int
  mlp_ratio: float = 4.0
  dropout_rate: float = 0.0
  attention_use_rope: bool = False
  attention_rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE
  normalization_type: normalization.NormalizationType = (
      normalization.NormalizationType.RMS_NORM
  )
  normalization_num_groups: int | None = None
  output_channels: int | None = None
  dtype: DType = jnp.float32

  def setup(self):
    self.patch_embedder = dit_blocks.PatchEmbedder(
        patch_size=self.patch_size,
        hidden_size=self.hidden_size,
        dtype=self.dtype,
    )

    self.norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=self.normalization_type,
        num_groups=self.normalization_num_groups,
        dtype=self.dtype,
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch height width channels"],
      conditioning_embeddings: dict[ConditioningMechanism, Float["batch ..."]],
      *,
      is_training: bool,
  ) -> Float["batch height width output_channels"]:
    b, h, w, input_channels = x.shape
    c = conditioning_embeddings.get(ConditioningMechanism.ADAPTIVE_NORM)
    if c is None:
      raise ValueError(
          "ADAPTIVE_NORM conditioning must be provided in"
          " conditioning_embeddings for DiT. Available keys in"
          f" conditioning_embeddings: {conditioning_embeddings.keys()}"
      )

    x_patches = self.patch_embedder(x)
    seq_dim = (h // self.patch_size) * (w // self.patch_size)
    assert x_patches.shape == (b, seq_dim, self.hidden_size)
    # x_patches is (B, L, D)

    # Add Positional Embeddings
    if not self.attention_use_rope:
      # Learnable positional embedding
      # We rely on the fact that if the shape changes, this might fail or
      # re-init if not handled carefully in JAX/Flax. Standard Flax usage
      # assumes fixed shape or careful handling. For this implementation, we
      # assume fixed max length or just current length.
      x_patches = sequence_embedders.AdditiveSequenceEmbedding(
          num_features=self.hidden_size
      )(x_patches)

    # Blocks
    for i in range(self.depth):
      x_patches = dit_blocks.DiTBlock(
          norm_factory=self.norm_factory,
          hidden_size=self.hidden_size,
          num_heads=self.num_heads,
          mlp_ratio=self.mlp_ratio,
          use_rope=self.attention_use_rope,
          rope_position_type=self.attention_rope_position_type,
          dtype=self.dtype,
          name=f"DiTBlock_{i}",
      )(x_patches, c, is_training=is_training)

    # Final Layer
    num_output_channels = (
        input_channels if self.output_channels is None else self.output_channels
    )

    x_out = dit_blocks.FinalLayer(
        norm_factory=self.norm_factory,
        patch_size=self.patch_size,
        out_channels=num_output_channels,
        dtype=self.dtype,
        name="FinalLayer",
    )(x_patches, c)

    x_out = utils.optional_bf16_to_fp32(x_out)
    return x_out
