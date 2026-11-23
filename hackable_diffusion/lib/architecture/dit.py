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

"""DiT backbone."""

from typing import TypeAlias

from flax import linen as nn
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.architecture import conditioning_encoder
from hackable_diffusion.lib.architecture import dit_blocks
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

Batch: TypeAlias = hd_api.Batch
ConditioningEmbeddings: TypeAlias = conditioning_encoder.ConditioningEmbeddings
DType: TypeAlias = jax_helpers.DType


################################################################################
# MARK: DiT
################################################################################


class DiT(nn.Module):
  """DiT model.

  A Diffusion Transformer backbone based on https://arxiv.org/abs/2212.09748.

  The conditioning type is determined by the block's `norm_strategy`. If
  `'padding_mask'` is present in `conditioning_embeddings`, it is forwarded as
  `mask` to each `DiTBlock`.

  Attributes:
    num_blocks: Number of DiT blocks.
    block: A DiTBlock module.
    encoder: Optional encoder module (e.g., Patchify for image inputs).
    decoder: Optional decoder module (e.g., DePatchify for image outputs).
    absolute_posenc: Optional absolute positional encoding module.
    cast_to_float32: Whether to cast the output to float32.
    dtype: Compute dtype forwarded to DiT blocks and conditional norms.
      Pre-built `encoder` and `decoder` modules use their own `dtype`.
  """

  num_blocks: int

  block: dit_blocks.DiTBlock

  encoder: nn.Module | None = None
  decoder: nn.Module | None = None

  absolute_posenc: nn.Module | None = None

  cast_to_float32: bool = False
  dtype: DType = jnp.float32

  def _compute_dtype(self) -> DType:
    return self.dtype if self.dtype != jnp.float32 else self.block.dtype

  def setup(self):
    compute_dtype = self._compute_dtype()
    norm_dtype = compute_dtype if compute_dtype != jnp.float32 else None
    self.conditional_norm = self.block.norm_strategy.build_layer(
        name='ConditionalNorm', dtype=norm_dtype
    )

  @nn.compact
  def __call__(
      self,
      x: hd_api.Data[Batch],
      conditioning_embeddings: ConditioningEmbeddings,
      *,
      is_training: bool,
  ) -> hd_api.Data[Batch]:
    adaptive_norm_emb = conditioning_embeddings.get(
        'adaptive_norm'
    )
    if adaptive_norm_emb is None:
      raise ValueError('adaptive_norm_emb must be provided.')

    padding_mask = conditioning_embeddings.get('padding_mask')
    compute_dtype = self._compute_dtype()

    # Tokenize the input and add positional encoding.
    tokens_emb = self.encoder(x) if self.encoder else x
    tokens_emb = jnp.asarray(tokens_emb, dtype=compute_dtype)
    if self.absolute_posenc:
      tokens_emb = tokens_emb + self.absolute_posenc(tokens_emb)

    # Apply DiT blocks.
    cond = jnp.asarray(adaptive_norm_emb, dtype=compute_dtype)
    for i in range(1, self.num_blocks + 1):
      tokens_emb = self.block.copy(name=f'Block_{i}', dtype=compute_dtype)(
          tokens_emb, cond, is_training=is_training, mask=padding_mask
      )

    # Apply final conditional norm.
    tokens_emb = self.conditional_norm(tokens_emb, c=nn.silu(cond))

    # Decode the tokens to the output.
    if self.decoder:
      tokens_emb = self.decoder(tokens_emb, cond)

    if self.cast_to_float32:
      tokens_emb = jax_helpers.optional_bf16_to_fp32(tokens_emb)
    return tokens_emb
