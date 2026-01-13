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

"""Backbones for simplicial data.

The assumption on simplicial data is slightly different from discrete data.
In the case of discrete data, the backbone satisfies the following:

  * Input [B, ..., 1] (index representation of token)
  * Output [B, ..., V] (logit output)

In the case of simplicial data, the backbone satisfies the following:

  * Input [B, ..., V] (input is a logit)
  * Output [B, ..., V] (output vocabulary)

Therefore our main modification is with respect to the embedding layer.
However, the projector can be shared with the discrete model.

Another important difference is that in the case of discrete data, `V` is the
vocabulary size, whereas in the case of simplicial data, `V` is the number of
simplex vertices. So, in practice, we have:

  * Discrete: `V = process.num_categories`
  * Simplicial: `V = process.process_num_categories`
"""

import abc
import einops
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import discrete
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float
Int = hd_typing.Int

ConditionalBackbone = arch_typing.ConditionalBackbone
ConditioningMechanism = arch_typing.ConditioningMechanism

BaseProjector = discrete.BaseProjector

################################################################################
# MARK: Probability Embedder
################################################################################


class BaseLogitEmbedder(nn.Module, abc.ABC):
  """Base class for probability embedders."""

  embedding_dim: int

  @abc.abstractmethod
  def __call__(
      self, x: Float['batch *other_input V'], is_training: bool
  ) -> Float['batch *other_embedding F']:
    ...


class DenseEmbedder(BaseLogitEmbedder):
  """Probability embedder that uses a dense layer.

  Attributes:
    adapt_to_image_like_data: Whether to adapt the simplex model to image-like
      data. If True, the input data is expected to have the shape <B, H, W, C,
      V>, which is embedded via `logit_embedder` in <B, H, W, C, F>, and then
      the last two dimensions are collapsed leading to <B, H, W, C*F>. This is
      then passed to the base backbone. After base backbone, the outputs are
      uncollapsed from <B, H, W, C*F>, back to <B, H, W, C, F>, and then
      projected to <B, H, W, C, V>  and the backbone outputs will be projected
      to the vocabulary size. If `False`, it simply uses logit_embedder,
      followed by the base backbone, and then projects to the vocabulary size.
    dtype: The dtype to use for the simplex model.
  """

  adapt_to_image_like_data: bool = False
  dtype: DType = jnp.float32

  @nn.compact
  @typechecked
  def __call__(
      self, x: Float['batch *other_input V'], is_training: bool
  ) -> Float['batch *other_embedding F']:
    del is_training  # Unused.

    x = jax.nn.softmax(x, axis=-1)
    # convert logits to probabilities

    logit_embeddings = nn.Dense(
        features=self.embedding_dim,
        dtype=self.dtype,
        name='logit_Embedding',
    )(x)

    # For image modality, we collapse channel dimension and probability
    # embeddings.
    if self.adapt_to_image_like_data:
      emb_dimension = self.embedding_dim
      channel_dimension = logit_embeddings.shape[-2]
      logit_embeddings = einops.rearrange(
          logit_embeddings,
          'b ... c f -> b ... (c f) ',
          f=emb_dimension,
          c=channel_dimension,
      )

    return logit_embeddings


################################################################################
# MARK: ConditionalSimplicialBackbone
################################################################################


class ConditionalSimplicialBackbone(ConditionalBackbone):
  """Conditional simplicial backbone for diffusion models.

  Attributes:
    base_backbone: The base backbone to use for the simplicial model. Can be any
      conditionl backbone such as MLP or UNet.
    logit_embedder: The probability embedder to use for the simplicial model.
    logit_projector: The probability projector to use for the simplicial model.
  """

  base_backbone: ConditionalBackbone
  logit_embedder: BaseLogitEmbedder
  logit_projector: BaseProjector

  def __post_init__(self):
    super().__post_init__()
    if self.logit_embedder.embedding_dim != self.logit_projector.embedding_dim:
      raise ValueError(
          'The embedding dimension of the probability embedder and the'
          ' probability projector must be the same. Got embedder embedding'
          f' dim: {self.logit_embedder.embedding_dim} and projector'
          f' embedding dim: {self.logit_projector.embedding_dim}.'
      )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Float['batch *other V'],
      conditioning_embeddings: dict[ConditioningMechanism, Float['batch ...']],
      is_training: bool,
  ) -> Float['batch *other V']:

    # Embed the probability distributions.
    logit_embeddings = self.logit_embedder(x, is_training=is_training)

    # Output the result of the base backbone.
    backbone_outputs = self.base_backbone(
        x=logit_embeddings,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )

    output = self.logit_projector(backbone_outputs, is_training=is_training)

    output = utils.optional_bf16_to_fp32(output)
    return output
