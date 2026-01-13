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

"""Backbones for discrete data."""

import abc
import einops
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float
Int = hd_typing.Int

ConditionalBackbone = arch_typing.ConditionalBackbone
ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Token Embedder
################################################################################


class BaseTokenEmbedder(nn.Module, abc.ABC):
  """Base class for token embedders."""

  embedding_dim: int

  @abc.abstractmethod
  def __call__(
      self, x: Int['batch *other_input 1'], is_training: bool
  ) -> Float['batch *other_embedding F']:
    ...


class TokenEmbedder(BaseTokenEmbedder):
  """Token embedder that uses a dense layer.

  Attributes:
    process_num_categories: The number of categories in the discrete process.
    adapt_to_image_like_data: Whether to adapt the discrete model to image-like
      data. If True, the input data is expected to have the shape <B, H, W, C,
      1>, which is then embedded via `token_embedder` into <B, H, W, C, F>, and
      then the last two dimensions are collapsed leading to <B, H, W, C*F>. This
      is then passed to the base backbone. After base backbone, the outputs are
      uncollapsed from <B, H, W, C*F>, back to <B, H, W, C, F>, and then
      projected to <B, H, W, C, V> and the backbone outputs will be projected
      to the vocabulary size. If `False`, it simply uses token_embedder,
      followed by the base backbone, and then projects to the vocabulary size.
    dtype: The dtype to use for the discrete model.
  """

  process_num_categories: int
  adapt_to_image_like_data: bool = False
  dtype: DType = jnp.float32

  @nn.compact
  @typechecked
  def __call__(
      self, x: Int['batch *other_input 1'], is_training: bool
  ) -> Float['batch *other_embedding F']:
    """Embeds the tokens into a hidden dimension.

    It assumes that `x` has a shape <B,...,1>, it then embeds it into
    `<B,...,F>`, where `F` is the embedding dimension, assuming that values of
    `x` lie in `[0, V-1]`, where `V` is the number of categories.

    Args:
      x: The tokens to embed.
      is_training: Whether the model is in training mode.
    """
    del is_training  # Unused.

    token_embeddings = nn.Embed(
        features=self.embedding_dim,
        num_embeddings=self.process_num_categories,
        dtype=self.dtype,
        name='Token_Embedding',
    )(x)
    token_embeddings = einops.rearrange(
        token_embeddings, 'b ... 1 c -> b ... c', c=self.embedding_dim
    )

    # For image modality, we collapse channel dimension and token embeddings.
    if self.adapt_to_image_like_data:
      emb_dimension = self.embedding_dim
      channel_dimension = token_embeddings.shape[-2]
      token_embeddings = einops.rearrange(
          token_embeddings,
          'b ... c f -> b ... (c f) ',
          f=emb_dimension,
          c=channel_dimension,
      )

    return token_embeddings


################################################################################
# MARK: Token Projector
################################################################################


class BaseProjector(nn.Module, abc.ABC):
  """Base class for projectors."""

  embedding_dim: int

  @abc.abstractmethod
  def __call__(
      self, x: Float['batch *other_embedding F'], is_training: bool
  ) -> Float['batch *other_input V']:

    ...


class DenseProjector(BaseProjector):
  """Projector that uses a dense layer.

  Attributes:
    num_categories: The vocabulary size of the model.
    adapt_to_image_like_data: Whether to adapt the model to image-like
      data. If True, the input data is expected to have the shape <B, H, W, C>
      (this means that the channel dimension has been collapsed), which is then
      projected and reshape into projected to <B, H, W, C, V>. If `False`, it
      simply uses token_projector, followed by the base backbone, and then
      projects to the vocabulary size. We refer to TokenEmbedder for more
      details.
    dtype: The dtype to use for the model.
  """

  num_categories: int
  adapt_to_image_like_data: bool = False
  dtype: DType = jnp.float32

  @nn.compact
  @typechecked
  def __call__(
      self, x: Float['batch *other_embedding F'], is_training: bool
  ) -> Float['batch *other_input V']:
    """Projects the token embeddings to the output vocabulary size."""
    del is_training  # Unused.

    if self.adapt_to_image_like_data:
      emb_dimension = self.embedding_dim
      channel_dimension = x.shape[-1] // self.embedding_dim
      x = einops.rearrange(
          x,
          'b ... (c f) -> b ... c f',
          c=channel_dimension,  # pytype: disable=name-error
          f=emb_dimension,  # pytype: disable=name-error
      )

    # Project the backbone outputs to to the output vocabulary size.
    output = nn.Dense(
        features=self.num_categories,
        dtype=self.dtype,
        name='Final_Projection',
    )(x)

    return output


################################################################################
# MARK: ConditionalDiscreteBackbone
################################################################################


class ConditionalDiscreteBackbone(ConditionalBackbone):
  """Conditional discrete backbone for diffusion models.

  Attributes:
    base_backbone: The base backbone to use for the discrete model. Can be any
      conditionl backbone such as MLP or UNet.
    token_embedder: The token embedder to use for the discrete model.
    token_projector: The token projector to use for the discrete model.
  """

  base_backbone: ConditionalBackbone
  token_embedder: BaseTokenEmbedder
  token_projector: BaseProjector

  def __post_init__(self):
    super().__post_init__()
    if self.token_embedder.embedding_dim != self.token_projector.embedding_dim:
      raise ValueError(
          'The embedding dimension of the token embedder and the token'
          ' projector must be the same. Got'
          f' embedder embedding dim: {self.token_embedder.embedding_dim} and'
          f' projector embedding dim: {self.token_projector.embedding_dim}.'
      )

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: Int['batch *other 1'],
      conditioning_embeddings: dict[ConditioningMechanism, Float['batch ...']],
      is_training: bool,
  ) -> Float['batch *other V']:

    # Embed the tokens.
    token_embeddings = self.token_embedder(x, is_training=is_training)

    # Output the result of the base backbone.
    backbone_outputs = self.base_backbone(
        x=token_embeddings,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )

    output = self.token_projector(backbone_outputs, is_training=is_training)

    output = utils.optional_bf16_to_fp32(output)
    return output
