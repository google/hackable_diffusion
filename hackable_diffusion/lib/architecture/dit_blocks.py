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

from typing import Any, Literal  # pylint: disable=unused-import
import dataclasses
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

RoPEPositionType = arch_typing.RoPEPositionType
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
# MARK: DiTBlock
################################################################################


class DiTBlock(nn.Module):
  """A DiT transformer block with configurable adaptive normalization.

  The conditioning behavior is fully determined by three inputs:

  - `norm_factory`: A `NormalizationLayerFactory` that configures the base
    normalization method (e.g. RMSNorm, LayerNorm) and the conditioning style
    (scale-only vs scale+shift).
  - `use_gates`: Whether to apply zero-init gates on residual branches.
  - `zero_init_output`: Whether to zero-init output projections in attention
    and FFN. Required to be True for identity-at-init when `use_gates=False`.

  Common configurations:

  - **Flux** (https://github.com/black-forest-labs/flux):

      norm_factory = NormalizationLayerFactory(
          normalization_method='rms_norm',
          use_conditional_shift=False,
      )
      use_gates = False
      zero_init_output = True

  - **SD3 / MMDiT** (https://arxiv.org/abs/2403.03206):

      norm_factory = NormalizationLayerFactory(
          normalization_method='rms_norm',
          use_scale=False,
          use_conditional_shift=True,
      )
      use_gates = True
      zero_init_output = False

  - **Original DiT** (https://arxiv.org/abs/2212.09748):

      norm_factory = NormalizationLayerFactory(
          normalization_method='layer_norm',
          use_bias=False,
          use_scale=False,
          use_conditional_shift=True,
      )
      use_gates = True
      zero_init_output = False

  Identity-at-init: each residual branch computes `x + branch(x)`. At init,
  the branch output must be zero so that the block acts as identity. This is
  achieved in one of two ways:

  - When `use_gates=True`: gates are zero-initialized, so
    `gate * branch(x) = 0` regardless of `branch(x)`.
  - When `use_gates=False`: the output projections in attention and FFN are
    zero-initialized (`zero_init_output=True`), so `branch(x) = 0` directly.

  Attributes:
    hidden_size: The hidden size of the block.
    num_heads: The number of attention heads.
    head_dim: The dimension of each attention head.
    norm_factory: Factory for creating conditional normalization layers.
    use_gates: Whether to use zero-init gates on residual branches.
    ffn_type: Feed-forward network type. 'swiglu' uses a gated SwiGLU FFN,
      'dense' uses a standard Dense layer.
    ffn_use_bias: Whether to use bias in the FFN. Modern DiT architectures tend
      to avoid bias in the FFN.
    ffn_activation: Activation function for the FFN.
    attn_normalize_qk: Whether to normalize query and key in attention.
    attn_use_bias: Whether to use bias in the attention QKV and output
      projections.
    mlp_ratio: The ratio of the MLP hidden dimension to the hidden size.
    use_rope: Whether to use RoPE.
    dropout_rate: The dropout rate.
    rope_position_type: The position type of RoPE.
    zero_init_output: Whether to zero-init output projections in attention and
      FFN. Required to be True for identity-at-init when `use_gates=False`.
    dtype: The dtype of the block.
  """

  hidden_size: int
  norm_factory: normalization.NormalizationLayerFactory
  num_heads: int = INVALID_INT
  head_dim: int = INVALID_INT
  use_gates: bool = True
  ffn_type: mlp_blocks.FFNType = 'swiglu'
  ffn_use_bias: bool = False
  ffn_activation: str = 'gelu'
  attn_normalize_qk: bool = True
  attn_use_bias: bool = True
  mlp_ratio: float = 4.0
  use_rope: bool = False
  dropout_rate: float = 0.0
  rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE
  zero_init_output: bool = True
  dtype: DType = jnp.float32

  def setup(self):
    if not self.mlp_ratio > 0:
      raise ValueError('MLP ratio must be positive.')
    mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
    if not mlp_hidden_dim > 0:
      raise ValueError('MLP hidden dimension must be positive.')

    if not self.use_gates and not self.zero_init_output:
      raise ValueError(
          'zero_init_output must be True when use_gates is False to ensure'
          ' identity-at-init.'
      )

    # FFN
    self.ffn = mlp_blocks.FeedForward(
        output_size=self.hidden_size,
        hidden_size=mlp_hidden_dim,
        ffn_type=self.ffn_type,
        zero_init_output=self.zero_init_output,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        activation=self.ffn_activation,
        use_bias=self.ffn_use_bias,
    )
    # Attention
    self.attn = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        use_rope=self.use_rope,
        rope_position_type=self.rope_position_type,
        zero_init_output=self.zero_init_output,
        dtype=self.dtype,
        normalize_qk=self.attn_normalize_qk,
        use_bias=self.attn_use_bias,
        dropout_rate=self.dropout_rate,
    )

    # Adaptive normalization.
    self.conditional_norm_attn = self.norm_factory.conditional_norm(
        norm_name='ConditionalNorm_Attention'
    )
    self.conditional_norm_mlp = self.norm_factory.conditional_norm(
        norm_name='ConditionalNorm_MLP'
    )

    # Gates.
    if self.use_gates:
      self.gate_msa = nn.Dense(
          self.hidden_size,
          kernel_init=nn.initializers.zeros_init(),
          bias_init=nn.initializers.zeros_init(),
          name='Dense_Gate_MSA',
      )
      self.gate_mlp = nn.Dense(
          self.hidden_size,
          kernel_init=nn.initializers.zeros_init(),
          bias_init=nn.initializers.zeros_init(),
          name='Dense_Gate_MLP',
      )

  @kt.typechecked
  @nn.compact
  def __call__(
      self,
      x: Float['*batch seq_dim emb_dim'],
      cond: Float['*#batch cond_dim'],
      *,
      is_training: bool,
      mask: Bool['batch seq_dim'] | None = None,
  ) -> Float['*batch seq_dim emb_dim']:
    """Calls the DiT block.

    Args:
      x: The input tensor.
      cond: The conditioning tensor.
      is_training: Whether the block is in training mode.
      mask: The self-attention padding mask.

    Returns:
      The output tensor.
    """
    cond_activated = nn.silu(cond)
    use_gates = self.use_gates

    # Attention Branch.
    x_normed = self.conditional_norm_attn(x, c=cond_activated)
    attn_out = self.attn(x_normed, c=None, mask=mask, is_training=is_training)
    if use_gates:
      gate_msa = self.gate_msa(cond_activated)
      attn_out = gate_msa[..., None, :] * attn_out
    x = x + attn_out

    # MLP Branch.
    x_normed = self.conditional_norm_mlp(x, c=cond_activated)
    mlp_out = self.ffn(x_normed, is_training=is_training)
    if use_gates:
      gate_mlp = self.gate_mlp(cond_activated)
      mlp_out = gate_mlp[..., None, :] * mlp_out
    x = x + mlp_out

    return x


################################################################################
# MARK: Custom DiT Blocks
################################################################################


class DiTBlockFlux(DiTBlock):
  """FLUX DiTBlock.

  Based on https://github.com/black-forest-labs/flux. It uses RMSNorm and
  zero-init output projections in attention and FFN to ensure identity-at-init.
  """

  norm_factory: normalization.NormalizationLayerFactory = dataclasses.field(
      init=False
  )
  use_gates: bool = dataclasses.field(init=False, default=False)
  zero_init_output: bool = dataclasses.field(init=False, default=True)
  attn_normalize_qk: bool = dataclasses.field(init=False, default=True)
  attn_use_bias: bool = dataclasses.field(init=False, default=False)

  def __post_init__(self):
    self.norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=arch_typing.NormalizationType.RMS_NORM,
        use_conditional_shift=False,
    )
    super().__post_init__()


class DiTBlockSD3(DiTBlock):
  """SD3 / MMDiT DiTBlock.

  Based on https://arxiv.org/abs/2403.03206. It uses RMSNorm with scale-only
  adaptive conditioning and zero-init gates on residual branches.
  """

  norm_factory: normalization.NormalizationLayerFactory = dataclasses.field(
      init=False
  )
  use_gates: bool = dataclasses.field(init=False, default=True)
  zero_init_output: bool = dataclasses.field(init=False, default=False)
  attn_normalize_qk: bool = dataclasses.field(init=False, default=True)
  attn_use_bias: bool = dataclasses.field(init=False, default=False)

  def __post_init__(self):
    self.norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=arch_typing.NormalizationType.RMS_NORM,
        use_scale=False,
        use_conditional_shift=True,
    )
    super().__post_init__()


class DiTBlockAdaLNZero(DiTBlock):
  """Original DiTBlock.

  Based on https://arxiv.org/abs/2212.09748. It uses LayerNorm with scale and
  shift adaptive conditioning and zero-init gates on residual branches.
  """

  norm_factory: normalization.NormalizationLayerFactory = dataclasses.field(
      init=False
  )
  use_gates: bool = dataclasses.field(init=False, default=True)
  zero_init_output: bool = dataclasses.field(init=False, default=False)
  attn_normalize_qk: bool = dataclasses.field(init=False, default=False)
  attn_use_bias: bool = dataclasses.field(init=False, default=True)

  def __post_init__(self):
    self.norm_factory = normalization.NormalizationLayerFactory(
        normalization_method=arch_typing.NormalizationType.LAYER_NORM,
        use_bias=False,
        use_scale=False,
        use_conditional_shift=True,
    )
    super().__post_init__()


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

  This module is a pure linear projection + reshape. Normalization should be
  applied before calling this module (e.g. in the DiT backbone).

  Attributes:
    patch_size: The size of the patches.
    output_shape: The shape of the output.
  """

  patch_size: tuple[int, int]
  output_shape: tuple[int, int, int]
  dtype: DType = jnp.float32

  @nn.compact
  @kt.typechecked
  def __call__(
      self,
      x: Float['*batch seq_dim emb_dim'],
      cond: Float['*#batch cond_dim'] | None = None,
  ) -> Float['*batch height width channels']:
    del cond  # Unused.
    h, w, c = self.output_shape
    hp, wp = self.patch_size
    hn = h // hp
    wn = w // wp

    x = nn.Dense(
        features=hp * wp * c,
        name='Dense_Out',
        dtype=self.dtype,
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
