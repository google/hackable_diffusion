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

from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import dit_blocks
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Constants
################################################################################

PAD_TOKEN = 0  # We use `0` because most text tokenizers use it for pad tokens.

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
Bool = hd_typing.Bool
DType = hd_typing.DType

DataArray = hd_typing.DataArray

ConditionalBackbone = arch_typing.ConditionalBackbone

################################################################################
# MARK: DiT
################################################################################


class DiT(nn.Module, ConditionalBackbone):
  """DiT model.

  A Diffusion Transformer backbone based on https://arxiv.org/abs/2212.09748.

  The conditioning type is determined by the block's `norm_factory`.

  Attributes:
    num_blocks: Number of DiT blocks.
    block: A DiTBlock module.
    encoder: Optional encoder module (e.g., Patchify for image inputs).
    decoder: Optional decoder module (e.g., DePatchify for image outputs).
    absolute_posenc: Optional absolute positional encoding module.
    cast_to_float32: Whether to cast the output to float32.
    dtype: The data type of the module.
    pad_token: The pad token value. This value is used in the attention function
      down the line to mask out padding tokens. Note that using this tokens
      assumes that the inputs to DiT are already tokenized, which is not the
      case for images.
    use_padding_mask: Whether to use a padding mask. Note that in this case we
      assume that inputs are already tokenized. By default, we are not using it
      because DiT is mainly used for image generation.
  """

  num_blocks: int

  block: dit_blocks.DiTBlock

  encoder: nn.Module | None = None
  decoder: nn.Module | None = None

  absolute_posenc: nn.Module | None = None

  cast_to_float32: bool = False
  dtype: DType = jnp.float32
  pad_token: int = PAD_TOKEN
  use_padding_mask: bool = False

  def setup(self):
    self.conditional_norm = self.block.norm_factory.conditional_norm()

  @kt.typechecked
  @nn.compact
  def __call__(
      self,
      x: DataArray,
      conditioning_embeddings: arch_typing.ConditioningEmbeddings,
      is_training: bool,
  ) -> DataArray:
    adaptive_norm_emb = conditioning_embeddings.get(
        'adaptive_norm'
    )
    if adaptive_norm_emb is None:
      raise ValueError("adaptive_norm_emb must be provided.")

    # TODO(agalashov): This assumes that x is already tokenized, which is not
    # true for images.
    if self.use_padding_mask:
      padding_mask = (x != self.pad_token).astype(jnp.bool_)
      padding_mask = jnp.squeeze(padding_mask, axis=-1)
    else:
      padding_mask = None

    # Tokenize the input and add positional encoding.
    tokens_emb = self.encoder(x) if self.encoder else x
    if self.absolute_posenc:
      tokens_emb = tokens_emb + self.absolute_posenc(tokens_emb)

    # Apply DiT blocks.
    cond = adaptive_norm_emb
    for i in range(1, self.num_blocks + 1):
      tokens_emb = self.block.copy(name=f"Block_{i}")(
          tokens_emb, cond, is_training=is_training, mask=padding_mask
      )

    # Apply final adaptive norm.
    tokens_emb = self.conditional_norm(tokens_emb, c=nn.silu(cond))

    # Decode the tokens to the output.
    if self.decoder:
      tokens_emb = self.decoder(tokens_emb, cond)

    if self.cast_to_float32:
      tokens_emb = jax_helpers.optional_bf16_to_fp32(tokens_emb)
    return tokens_emb
