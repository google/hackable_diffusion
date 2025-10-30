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

"""Attention layers and utils."""

from typing import Callable
import warnings

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import sequence_embedders
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp


################################################################################
# MARK: Type aliases
################################################################################

Float = hd_typing.Float
DType = hd_typing.DType

RoPEPositionType = arch_typing.RoPEPositionType
INVALID_INT = arch_typing.INVALID_INT

################################################################################
# MARK: Constants
################################################################################

SAFETY_EPSILON = 1e-6

################################################################################
# MARK: Attention utilities
################################################################################


def attention_dims_factory(
    head_dim: int, num_heads: int
) -> Callable[[Float["batch sequence dim"]], tuple[int, int]]:
  """Returns a function that returns the head dimension and number of heads."""

  if head_dim != INVALID_INT and head_dim <= 0:
    raise ValueError("Head dimension must be positive or INVALID_INT.")
  elif num_heads != INVALID_INT and num_heads <= 0:
    raise ValueError("Number of heads must be positive or INVALID_INT.")

  if head_dim == INVALID_INT and num_heads == INVALID_INT:
    raise ValueError("Either head_dim or num_heads must be specified.")
  elif head_dim != INVALID_INT and num_heads == INVALID_INT:

    def get_attention_dims(
        x: Float["batch sequence dim"],
    ) -> tuple[int, int]:
      *_, d = x.shape  # batch size, sequence length, embedding dim

      if d % head_dim != 0:
        raise ValueError(
            f"Embedding dim {d} is not divisible by head_dim {head_dim}."
        )

      num_heads = d // head_dim
      return head_dim, num_heads

    return get_attention_dims

  elif head_dim == INVALID_INT and num_heads != INVALID_INT:

    def get_attention_dims(
        x: Float["batch sequence dim"],
    ) -> tuple[int, int]:
      *_, d = x.shape  # batch size, sequence length
      if d % num_heads != 0:
        raise ValueError(
            f"Embedding dim {d} is not divisible by num_heads {num_heads}."
        )
      head_dim = d // num_heads
      return head_dim, num_heads

    return get_attention_dims

  else:
    raise ValueError("Either head_dim or num_heads must be INVALID_INT.")


@typechecked
def _stable_softmax(
    logits: Float["*sequence dim"],
) -> Float["*sequence dim"]:
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
        "Softmax expects logits in float32 or bfloat16, but got %s",
        logits.dtype,
    )
    logits = logits.astype(jnp.float32)
    output = jax.nn.softmax(logits)
  return output


@typechecked
def _dot_product_attention(
    q: Float["batch head sequence_query dim"],
    k: Float["batch head sequence_key dim"],
    v: Float["batch head sequence_key dim"],
    rescale: Float["..."],
) -> Float["batch sequence_query head*dim"]:
  """Dot product attention."""

  b, _, t, _ = q.shape

  # Attention scores
  attn_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * rescale

  # Softmax and attention weights
  attn_weights = _stable_softmax(logits=attn_logits)

  # Calculate attention output
  attn_output = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)

  # Merge heads and project to output dimension
  attn_output = attn_output.transpose(0, 2, 1, 3).reshape(b, t, -1)

  return attn_output


################################################################################
# MARK: Multi-head attention
################################################################################


class MultiHeadAttention(nn.Module):
  """Multi-head attention layer.

  This module implements multi-head attention, supporting both self-attention
  and cross-attention. If conditioning `c` is provided, cross-attention is
  performed using `x` as query and `c` as key/value. Otherwise, self-attention
  is performed using `x` as query, key, and value.

  It supports RoPE for positional embeddings and QK normalization.

  Attributes:
    num_heads: The number of attention heads. If set to INVALID_INT, it is
      inferred from head_dim and input channels.
    head_dim: The dimension of each attention head. If set to INVALID_INT, it is
      inferred from num_heads and input channels. One of num_heads or head_dim
      must be INVALID_INT.
    normalize_qk: Whether to normalize query and key before attention.
    use_rope: Whether to use rotary positional embeddings on query and key.
    rope_position_type: The type of rotary positional embeddings to use if
      use_rope is True.
    zero_init_output: If True, the kernel of the final output projection layer
      is initialized to zeros.
    dtype: The data type of the computation.
  """

  num_heads: int = INVALID_INT
  head_dim: int = INVALID_INT
  normalize_qk: bool = False
  use_rope: bool = False
  rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE
  zero_init_output: bool = False
  dtype: DType = jnp.float32

  def setup(self):
    self.init_q = nn.linear.default_kernel_init
    self.init_k = nn.linear.default_kernel_init
    self.init_v = nn.linear.default_kernel_init
    if self.zero_init_output:
      self.init_output = nn.initializers.zeros_init()
    else:
      self.init_output = nn.linear.default_kernel_init

    self.get_attention_dims = attention_dims_factory(
        head_dim=self.head_dim, num_heads=self.num_heads
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float["batch sequence1 dim1"],
      c: Float["batch sequence2 dim2"] | None,
  ) -> Float["batch sequence1 dim1"]:
    b, _, d = x.shape  # batch size, sequence length, embedding dim
    head_d, num_heads = self.get_attention_dims(x)

    # if c is None, use x (self-attention)
    y = x if c is None else c
    seq_len_kv = y.shape[1]
    seq_len_q = x.shape[1]

    q = nn.Dense(
        features=d,
        kernel_init=self.init_q,
        dtype=self.dtype,
        name="Dense_Q",
    )(x)
    k = nn.Dense(
        features=d,
        kernel_init=self.init_k,
        dtype=self.dtype,
        name="Dense_K",
    )(y)
    v = nn.Dense(
        features=d,
        kernel_init=self.init_v,
        dtype=self.dtype,
        name="Dense_V",
    )(y)

    # Reshape to multiple heads
    q = q.reshape(b, seq_len_q, num_heads, head_d).transpose(0, 2, 1, 3)
    k = k.reshape(b, seq_len_kv, num_heads, head_d).transpose(0, 2, 1, 3)
    v = v.reshape(b, seq_len_kv, num_heads, head_d).transpose(0, 2, 1, 3)
    # shape is [batch, num_heads, sequence_length, head_dim]

    # RoPE: https://arxiv.org/abs/2104.09864
    if self.use_rope:
      q = sequence_embedders.RoPESequenceEmbedding(
          rope_position_type=self.rope_position_type
      )(q)
      k = sequence_embedders.RoPESequenceEmbedding(
          rope_position_type=self.rope_position_type
      )(k)
      # shape is [batch, num_heads, sequence_length, head_dim]

    # QK normalization: https://arxiv.org/abs/2010.04245.
    if self.normalize_qk:
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
      scale = 1.0 / jnp.sqrt(head_d)

    attn_output = _dot_product_attention(
        q=q,
        k=k,
        v=v,
        rescale=scale,
    )

    attn_output = nn.Dense(
        features=d,
        kernel_init=self.init_output,
        dtype=self.dtype,
        name="Dense_Output",
    )(attn_output)

    return attn_output
