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

"""Attention layers and utils."""

import dataclasses
from typing import Literal
import warnings

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import sequence_embedders
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
Bool = hd_typing.Bool
DType = hd_typing.DType

RoPEPositionsFn = sequence_embedders.RoPEPositionsFn
SquareRoPEPositions = sequence_embedders.SquareRoPEPositions

AttnQKNormMethod = Literal["l2", "rms_norm"]

################################################################################
# MARK: Constants
################################################################################

SAFETY_EPSILON = 1e-6
MASK_LOGITS_VALUE = -1e9

################################################################################
# MARK: Attention Utilities
################################################################################


@dataclasses.dataclass(frozen=True)
class AttentionHeadsSpec:
  """Configuration for multi-head attention dimensionality.

  Specify at least one of `num_heads` or `head_dim`. When only one is set, the
  other is inferred from the embedding dimension at call time via `resolve()`.
  When both are set, `resolve()` validates that the embedding dimension matches
  `num_heads * head_dim`.

  Attributes:
    num_heads: Fixed number of attention heads.
    head_dim: Fixed dimension per head.
  """

  num_heads: int | None = None
  head_dim: int | None = None

  def __post_init__(self):
    if not self.num_heads and not self.head_dim:
      raise ValueError("At least num_heads or head_dim must be provided.")

  def resolve(self, embedding_dim: int) -> tuple[int, int]:
    """Returns (head_dim, num_heads) given the embedding dimension."""
    if self.head_dim and self.num_heads:
      expected = self.head_dim * self.num_heads
      if embedding_dim != expected:
        raise ValueError(
            f"Expected embedding_dim={expected}, got {embedding_dim}."
            f" (head_dim={self.head_dim} * num_heads={self.num_heads})"
        )
      return self.head_dim, self.num_heads

    if self.head_dim:
      if embedding_dim % self.head_dim != 0:
        raise ValueError(
            f"Embedding dim {embedding_dim} is not divisible by"
            f" head_dim {self.head_dim}."
        )
      return self.head_dim, embedding_dim // self.head_dim

    if embedding_dim % self.num_heads != 0:  # pyrefly: ignore[unsupported-operation]
      raise ValueError(
          f"Embedding dim {embedding_dim} is not divisible by"
          f" num_heads {self.num_heads}."
      )
    return embedding_dim // self.num_heads, self.num_heads  # pyrefly: ignore[bad-return, unsupported-operation]


@kt.typechecked
def _stable_softmax(
    logits: Float["*sequence dim"],  # pyrefly: ignore[not-a-type]
) -> Float["*sequence dim"]:  # pyrefly: ignore[not-a-type]
  """Numerically stable softmax for (potential) bfloat 16."""
  if logits.dtype == jnp.float32:
    output = jax.nn.softmax(logits)
  elif logits.dtype == jnp.bfloat16:
    # Need to explicitly do softmax in float32 to avoid numerical issues
    # with large negatives. Large negatives can occur if trying to mask
    # by adding on large negative logits so that things softmax to zero.
    output = jax.nn.softmax(logits.astype(jnp.float32)).astype(jnp.bfloat16)
  else:
    warnings.warn(
        "Softmax expects logits in float32 or bfloat16, but got"
        f" {logits.dtype}",
    )
    logits = logits.astype(jnp.float32)
    output = jax.nn.softmax(logits)
  return output


@kt.typechecked
def _dot_product_attention(
    q: Float["batch head sequence_query dim"],  # pyrefly: ignore[not-a-type]
    k: Float["batch head sequence_key dim"],  # pyrefly: ignore[not-a-type]
    v: Float["batch head sequence_key dim"],  # pyrefly: ignore[not-a-type]
    rescale: Float["..."],  # pyrefly: ignore[bad-index, not-a-type]
    *,
    mask: Bool["batch sequence_key"] | None = None,
    dropout_rate: float = 0.0,
    is_training: bool = True,
) -> Float["batch sequence_query head*dim"]:  # pyrefly: ignore[not-a-type]
  """Performs dot product attention.

  Args:
    q: Query tensor.
    k: Key tensor.
    v: Value tensor.
    rescale: Rescale factor for the attention scores.
    mask: Mask tensor. Mask is True for tokens we want to keep and False for
      tokens we want to mask. If None, no masking is performed.
    dropout_rate: The dropout rate for the attention weights.
    is_training: Whether the model is in training mode.

  Returns:
    The output tensor.
  """

  b, _, t, _ = q.shape

  # Attention scores
  attention_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * rescale

  # We apply the mask to the logits before softmax so that the softmax is zero
  # for masked tokens.
  if mask is not None:
    bcast_mask = jnp.expand_dims(mask, axis=(1, 2))
    attention_logits = jnp.where(bcast_mask, attention_logits, MASK_LOGITS_VALUE)

  # Softmax and attention weights
  attention_weights = _stable_softmax(logits=attention_logits)

  if dropout_rate > 0.0:
    attention_weights = nn.Dropout(rate=dropout_rate)(
        attention_weights, deterministic=not is_training
    )

  # Calculate attention output
  attention_output = jnp.einsum("bhts,bhsd->bhtd", attention_weights, v)

  # Merge heads and project to output dimension
  attention_output = attention_output.transpose(0, 2, 1, 3).reshape(b, t, -1)

  return attention_output


################################################################################
# MARK: Multi-Head Attention
################################################################################


class MultiHeadAttention(nn.Module):
  """Multi-head attention layer.

  This module implements multi-head attention, supporting both self-attention
  and cross-attention. If conditioning `c` is provided, cross-attention is
  performed using `x` as query and `c` as key/value. Otherwise, self-attention
  is performed using `x` as query, key, and value. Additionally, a mask can be
  provided to mask out certain tokens, such as padding tokens. Mask is True for
  tokens that should be kept and False for tokens that should be masked.

  It supports RoPE for positional embeddings and QK normalization.

  Attributes:
    attention_heads_spec: An AttentionHeadsSpec instance that specifies num_heads
      and/or head_dim for the attention layer.
    normalize_qk: Whether to normalize query and key before attention.
    use_rope: Whether to use rotary positional embeddings on query and key.
    rope_positions_fn: The position function of rotary positional embeddings to
      use if use_rope is True.
    use_bias: Whether to use bias in the QKV and output projections.
    zero_init_output: If True, the kernel of the final output projection layer
      is initialized to zeros.
    dropout_rate: The dropout rate for the attention weights.
    dtype: The data type of the computation.
  """

  attention_heads_spec: AttentionHeadsSpec
  normalize_qk: bool = False
  qk_norm_method: AttnQKNormMethod = "l2"
  use_rope: bool = False
  rope_positions_fn: RoPEPositionsFn = SquareRoPEPositions()
  use_bias: bool = True
  zero_init_output: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32

  def setup(self):
    self.init_q = nn.linear.default_kernel_init
    self.init_k = nn.linear.default_kernel_init
    self.init_v = nn.linear.default_kernel_init
    if self.zero_init_output:
      self.init_output = nn.initializers.zeros_init()
    else:
      self.init_output = nn.linear.default_kernel_init

  @nn.compact
  @kt.typechecked
  def __call__(
      self,
      x: Float["batch sequence1 dim1"],  # pyrefly: ignore[not-a-type]
      c: Float["batch sequence2 dim2"] | None,
      *,
      mask: Bool["batch sequence1|sequence2"] | None = None,
      is_training: bool = True,
  ) -> Float["batch sequence1 dim1"]:  # pyrefly: ignore[not-a-type]
    """Computes multi-head attention.

    Args:
      x: The input tensor.
      c: The conditioning tensor.
      mask: The mask tensor. Mask is True for tokens we want to keep and False
        for tokens we want to mask. If None, no masking is performed. In
        cross-attention, mask is applied to the key/value sequence which comes
        from c. In self-attention, the mask is applied to x.
      is_training: Whether the model is in training mode (for dropout)

    Returns:
      The output tensor.
    """
    if c is None:
      if mask is not None and mask.shape != x.shape[:2]:
        raise ValueError(
            f"In self-attention, mask shape {mask.shape} does not match"
            f" expected shape {x.shape[:2]}."
        )
    else:
      if mask is not None and mask.shape != c.shape[:2]:
        raise ValueError(
            f"In cross-attention, mask shape {mask.shape} does not match"
            f" expected shape {c.shape[:2]}."
        )
    b, _, d = x.shape  # batch size, sequence length, embedding dim
    head_d, num_heads = self.attention_heads_spec.resolve(d)

    # if c is None, use x (self-attention)
    y = x if c is None else c

    seq_len_kv = y.shape[1]
    seq_len_q = x.shape[1]

    q = nn.Dense(
        features=d,
        use_bias=self.use_bias,
        kernel_init=self.init_q,
        dtype=self.dtype,
        name="Dense_Q",
    )(x)
    k = nn.Dense(
        features=d,
        use_bias=self.use_bias,
        kernel_init=self.init_k,
        dtype=self.dtype,
        name="Dense_K",
    )(y)
    v = nn.Dense(
        features=d,
        use_bias=self.use_bias,
        kernel_init=self.init_v,
        dtype=self.dtype,
        name="Dense_V",
    )(y)

    # Reshape to multiple heads
    q = q.reshape(b, seq_len_q, num_heads, head_d).transpose(0, 2, 1, 3)
    k = k.reshape(b, seq_len_kv, num_heads, head_d).transpose(0, 2, 1, 3)
    v = v.reshape(b, seq_len_kv, num_heads, head_d).transpose(0, 2, 1, 3)
    # shape is [batch, num_heads, sequence_length, head_dim]

    # QK normalization: https://arxiv.org/abs/2010.04245.
    if self.normalize_qk:
      if self.qk_norm_method == "rms_norm":
        q = nn.RMSNorm(name="RMSNorm_Q")(q)
        k = nn.RMSNorm(name="RMSNorm_K")(k)
        scale = 1.0 / jnp.sqrt(jnp.float32(head_d))
      # QK L2 normalization: https://arxiv.org/abs/2010.04245
      elif self.qk_norm_method == "l2":
        scale = self.param(
            "norm_qk_scale",
            nn.initializers.constant(
                jnp.log2(seq_len_kv**2 - seq_len_kv + SAFETY_EPSILON)
            ),
            (1, 1, 1, 1),
        )

        norm_q = jnp.linalg.norm(q, ord=2, axis=-1, keepdims=True)
        norm_k = jnp.linalg.norm(k, ord=2, axis=-1, keepdims=True)
        q = q / (norm_q + SAFETY_EPSILON)
        k = k / (norm_k + SAFETY_EPSILON)
      else:
        raise ValueError(
            f"Unsupported QK normalization method: {self.qk_norm_method}."
        )
    else:
      scale = 1.0 / jnp.sqrt(jnp.float32(head_d))

    # RoPE: https://arxiv.org/abs/2104.09864
    if self.use_rope:
      q = sequence_embedders.RoPESequenceEmbedding(
          rope_positions_fn=self.rope_positions_fn
      )(q)
      k = sequence_embedders.RoPESequenceEmbedding(
          rope_positions_fn=self.rope_positions_fn
      )(k)
      # shape is [batch, num_heads, sequence_length, head_dim]

    attention_output = _dot_product_attention(
        q=q,
        k=k,
        v=v,
        rescale=scale,
        mask=mask,
        dropout_rate=self.dropout_rate,
        is_training=is_training,
    )

    attention_output = nn.Dense(
        features=d,
        use_bias=self.use_bias,
        kernel_init=self.init_output,
        dtype=self.dtype,
        name="Dense_Output",
    )(attention_output)

    return attention_output
