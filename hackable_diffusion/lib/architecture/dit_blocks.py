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

"""DiT building blocks."""

import einops
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import mlp_blocks
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp

################################################################################
# MARK: Type aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

NormalizationLayerFactory = normalization.NormalizationLayerFactory
RoPEPositionType = arch_typing.RoPEPositionType


################################################################################
# MARK: Patch Embedder
################################################################################


class PatchEmbedder(nn.Module):
  """Patch embedding layer.

  Splits the image into patches and embeds them.

  Attributes:
    patch_size: The size of the patches.
    hidden_size: The dimension of the embedding.
    dtype: The data type of the computation.
  """

  patch_size: int
  hidden_size: int
  dtype: DType = jnp.float32

  @nn.compact
  @typechecked
  def __call__(
      self, x: Float["batch height width channels"]
  ) -> Float["batch sequence hidden_size"]:
    b, h, w, _ = x.shape
    if h % self.patch_size != 0 or w % self.patch_size != 0:
      raise ValueError(
          f"Image dimensions ({h}, {w}) must be divisible by patch size"
          f" ({self.patch_size})."
      )

    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding="VALID",
        dtype=self.dtype,
        name="PatchEmbedder_Conv",
    )(x)
    # x is now (B, H//P, W//P, D)
    assert x.shape == (
        b,
        h // self.patch_size,
        w // self.patch_size,
        self.hidden_size,
    )

    return x.reshape(b, -1, self.hidden_size)


################################################################################
# MARK: DiT Block
################################################################################


class DiTBlock(nn.Module):
  """Diffusion Transformer Block.

  Attributes:
    norm_factory: Factory for creating normalization layers.
    hidden_size: The dimension of the hidden state.
    num_heads: The number of attention heads.
    mlp_ratio: The ratio of the hidden dimension in the MLP to the input
      dimension.
    use_rope: Whether to use rotary positional embeddings.
    rope_position_type: The type of rotary positional embeddings.
    dtype: The data type of the computation.
  """

  norm_factory: NormalizationLayerFactory
  hidden_size: int
  num_heads: int
  mlp_ratio: float = 4.0
  use_rope: bool = False
  rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE
  dtype: DType = jnp.float32

  def setup(self):
    if not self.mlp_ratio > 0:
      raise ValueError("MLP ratio must be positive.")
    mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
    if not mlp_hidden_dim > 0:
      raise ValueError("MLP hidden dimension must be positive.")

    self.norm = self.norm_factory.conditional_norm_factory()
    self.attn = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        use_rope=self.use_rope,
        rope_position_type=self.rope_position_type,
        zero_init_output=True,
        dtype=self.dtype,
    )
    # self.norm2 = self.norm_factory.conditional_norm_factory()
    self.mlp = mlp_blocks.MLP(
        hidden_sizes=[mlp_hidden_dim],
        output_size=self.hidden_size,
        activation="gelu",
        activate_final=False,
        zero_init_output=True,
        dtype=self.dtype,
        name="MLP",
    )
    # Part of AdaLNZero architecture
    # (scale/shift taken care of by the Conditional Normalizations)
    self.attn_gate = nn.Dense(
        features=self.hidden_size,
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        dtype=self.dtype,
        name="AttnGate",
    )
    self.mlp_gate = nn.Dense(
        features=self.hidden_size,
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        dtype=self.dtype,
        name="MLPGate",
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch sequence hidden_size"],
      c: Float["batch cond_dim"],
      *,
      is_training: bool,
  ) -> Float["batch sequence hidden_size"]:
    pad_to_seq_axis = lambda t: einops.rearrange(
        utils.bcast_right(einops.rearrange(t, "b ... c -> b c ..."), x.ndim),
        "b c ... -> b ... c",
    )
    attn_gate = pad_to_seq_axis(self.attn_gate(c))
    mlp_gate = pad_to_seq_axis(self.mlp_gate(c))
    x = x + attn_gate * self.attn(self.norm(x, c), c=None)

    # Re-use MLP which assumes (B, Dim) structure.
    def _seq_mlp(x):
      b, t, d = x.shape
      x = jnp.reshape(x, (b * t, d))
      y = self.mlp(x, is_training=is_training)
      y = y.reshape(b, t, d)
      return y

    return x + mlp_gate * _seq_mlp(self.norm(x, c))


################################################################################
# MARK: Final Layer
################################################################################


class FinalLayer(nn.Module):
  """Final layer of DiT.

  Attributes:
    norm_factory: Factory for creating normalization layers.
    patch_size: The size of the patches.
    out_channels: The number of output channels.
    dtype: The data type of the computation.
  """

  norm_factory: NormalizationLayerFactory
  patch_size: int
  out_channels: int
  dtype: DType = jnp.float32

  def setup(self):
    self.norm_final = self.norm_factory.conditional_norm_factory()
    self.linear = nn.Dense(
        features=self.patch_size * self.patch_size * self.out_channels,
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        dtype=self.dtype,
        name="Final_Linear",
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch sequence hidden_size"],
      c: Float["batch cond_dim"],
  ) -> Float["batch height width out_channels"]:
    x = self.norm_final(x, c)
    x = self.linear(x)

    # Unpatchify, assuming square image (and square patches)
    b, l, _ = x.shape
    h = w = int(jnp.sqrt(l))
    if (h * w) != l:
      raise ValueError(
          f"Number of patches ({h}x{w}) is not divisible by sequence length"
          f" ({l})."
      )

    # x is (B, H*W, P*P*C)
    x = x.reshape(b, h, w, self.patch_size, self.patch_size, self.out_channels)
    # (B, H, W, P, P, C) -> (B, H, P, W, P, C) -> (B, H*P, W*P, C)
    x = jnp.einsum("bhwpqc->bhpwqc", x)
    x = x.reshape(
        b, h * self.patch_size, w * self.patch_size, self.out_channels
    )
    return x
