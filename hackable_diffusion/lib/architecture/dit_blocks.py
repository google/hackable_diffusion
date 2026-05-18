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
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import mlp_blocks
from hackable_diffusion.lib.architecture import normalization
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float
Bool = hd_typing.Bool
Num = hd_typing.Num

NormalizationLayer = normalization.NormalizationLayer
RoPEPositionType = arch_typing.RoPEPositionType
NormalizationType = arch_typing.NormalizationType
INVALID_INT = arch_typing.INVALID_INT


################################################################################
# MARK: PositionalEmbedding
################################################################################


class PositionalEmbedding(nn.Module):
  """Learnable additive sequence positional embedding."""

  init_stddev: float = 0.02

  @nn.compact
  @kt.typechecked
  def __call__(self, x: Num["batch *data_shape"]) -> Float["batch *data_shape"]:
    pos_embed = self.param(
        "PositionalEmbeddingTensor",
        nn.initializers.normal(stddev=self.init_stddev),
        (1, *x.shape[1:]),
        x.dtype,
    )

    # Broadcast (1, seq, dim) to (batch, seq, dim) to satisfy jaxtyping
    return jnp.broadcast_to(pos_embed, x.shape)


################################################################################
# MARK: DiTBlockAdaLNZero
################################################################################


class DiTBlockAdaLNZero(nn.Module):
  """A DiT block with a single unified adaLN-Zero projection.

  Attributes:
    hidden_size: The hidden size of the block.
    num_heads: The number of attention heads.
    head_dim: The dimension of each attention head.
    mlp_ratio: The ratio of the MLP hidden dimension to the hidden size.
    use_rope: Whether to use RoPE.
    rope_position_type: The position type of RoPE.
    dtype: The dtype of the block.
  """

  hidden_size: int
  num_heads: int = INVALID_INT
  head_dim: int = INVALID_INT
  mlp_ratio: float = 4.0
  use_rope: bool = False
  dropout_rate: float = 0.0
  rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE
  dtype: DType = jnp.float32

  def setup(self):
    if not self.mlp_ratio > 0:
      raise ValueError("MLP ratio must be positive.")
    mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
    if not mlp_hidden_dim > 0:
      raise ValueError("MLP hidden dimension must be positive.")

    self.mlp = mlp_blocks.MLP(
        hidden_sizes=[mlp_hidden_dim],
        output_size=self.hidden_size,
        activation="gelu",
        activate_final=False,
        zero_init_output=False,
        dtype=self.dtype,
        name="MLP",
    )
    self.attn = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        use_rope=self.use_rope,
        rope_position_type=self.rope_position_type,
        zero_init_output=False,
        dtype=self.dtype,
        normalize_qk=True,
    )
    self.gate_msa = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        name="Dense_Gate_MSA",
    )
    self.gate_mlp = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(),
        name="Dense_Gate_MLP",
    )
    self.conditional_norm_attn = normalization.NormalizationLayerFactory(
        normalization_method=NormalizationType.LAYER_NORM,
        dtype=self.dtype,
        use_bias=False,
        use_scale=False,
    ).conditional_norm(norm_name="ConditionalNorm_Attention")
    self.conditional_norm_mlp = normalization.NormalizationLayerFactory(
        normalization_method=NormalizationType.LAYER_NORM,
        dtype=self.dtype,
        use_bias=False,
        use_scale=False,
    ).conditional_norm(norm_name="ConditionalNorm_MLP")

  @kt.typechecked
  @nn.compact
  def __call__(
      self,
      x: Float["*batch seq_dim emb_dim"],
      cond: Float["*#batch cond_dim"],
      *,
      is_training: bool,
      mask: Bool["batch seq_dim"] | None = None,
  ) -> Float["*batch seq_dim emb_dim"]:
    """Calls the DiT block.

    Args:
      x: The input tensor.
      cond: The conditioning tensor.
      is_training: Whether the block is in training mode.
      mask: The self-attention padding mask. If the mask is provided, it is
        assumed that the input sequence contains padding tokens that should be
        masked out when computing the self-attention.

    Returns:
      The output tensor.
    """

    # Attention Branch
    x_attn_modulated = self.conditional_norm_attn(x, c=nn.silu(cond))
    attn_out = self.attn(x_attn_modulated, c=None, mask=mask)
    # Optional dropout
    if self.dropout_rate > 0.0:
      attn_out = nn.Dropout(rate=self.dropout_rate)(
          attn_out, deterministic=not is_training
      )
    gate_msa = self.gate_msa(nn.silu(cond))
    # Add a sequence dimension [...,None,:] to broadcast to [*batch,seq,dim].
    x = x + gate_msa[..., None, :] * attn_out

    # MLP Branch
    x_mlp_modulated = self.conditional_norm_mlp(x, c=nn.silu(cond))
    mlp_out = self.mlp(x_mlp_modulated, is_training=is_training)
    # Optional dropout
    if self.dropout_rate > 0.0:
      mlp_out = nn.Dropout(rate=self.dropout_rate)(
          mlp_out, deterministic=not is_training
      )
    gate_mlp = self.gate_mlp(nn.silu(cond))
    # Add a sequence dimension [...,None,:] to broadcast to [*batch,seq,dim].
    x = x + gate_mlp[..., None, :] * mlp_out
    return x


################################################################################
# MARK: Patchify
################################################################################


class Patchify(nn.Module):
  """Patchify module.

  Flattens the input image into a sequence of patches and projects them to a
  sequence of embeddings.

  Attributes:
    patch_size: The size of the patches.
    embedding_dim: The dimension of the embedding.
  """

  patch_size: tuple[int, int]
  embedding_dim: int

  @nn.compact
  @kt.typechecked
  def __call__(
      self, x: Float["*batch height width channels"]
  ) -> Float["*batch seq_dim emb_dim"]:
    hp, wp = self.patch_size
    _, h, w, _ = x.shape
    if h % hp != 0 or w % wp != 0:
      raise ValueError(
          f"Height {h} must be divisible by patch height {hp}."
          f"Width {w} must be divisible by patch width {wp}."
      )

    reshaped = einops.rearrange(
        x,
        "... (hn hp) (wn wp) c -> ... (hn wn) (hp wp c)",
        hp=hp,
        wp=wp,
    )
    reshaped = nn.Dense(features=self.embedding_dim, name="Dense_Project")(
        reshaped
    )
    return reshaped


################################################################################
# MARK: DePatchify
################################################################################


class DePatchify(nn.Module):
  """DePatchify module.

  Works in the opposite direction of Patchify. Projects a sequence of embeddings
  to a sequence of patches. Then reshapes the sequence of patches to the
  original image shape.

  Attributes:
    patch_size: The size of the patches.
    output_shape: The shape of the output.
  """

  patch_size: tuple[int, int]
  output_shape: tuple[int, int, int]
  dtype: DType = jnp.float32

  def setup(self):
    self.conditional_norm = normalization.NormalizationLayerFactory(
        normalization_method=NormalizationType.LAYER_NORM,
        dtype=self.dtype,
        use_bias=False,
        use_scale=False,
    ).conditional_norm()

  @nn.compact
  @kt.typechecked
  def __call__(
      self, x: Float["*batch seq_dim emb_dim"], cond: Float["*#batch cond_dim"]
  ) -> Float["*batch height width channels"]:
    h, w, c = self.output_shape
    hp, wp = self.patch_size
    hn = h // hp
    wn = w // wp

    x = self.conditional_norm(x, c=nn.silu(cond))
    x = nn.Dense(
        features=hp * wp * c,
        name="Dense_Out",
    )(x)

    return einops.rearrange(
        x,
        "... (hn wn) (hp wp c) -> ... (hn hp) (wn wp) c",
        hn=hn,
        wn=wn,
        hp=hp,
        wp=wp,
        c=c,
    )
