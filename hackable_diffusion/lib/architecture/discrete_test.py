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

"""Tests for discrete backbones."""

import itertools

from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import discrete
from hackable_diffusion.lib.architecture import mlp
from hackable_diffusion.lib.architecture import unet
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Type Aliases
################################################################################

ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Tests
################################################################################


class ConditionalDiscreteBackboneTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.is_training = True
    self.shape = (4, 4, 3, 1)
    self.prod_shape = int(np.prod(self.shape))
    self.cond_dim = 16
    self.discrete_x = jnp.ones((self.batch_size, *self.shape), dtype=jnp.int32)
    self.concatenate_emb = {
        ConditioningMechanism.CONCATENATE: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    self.sum_emb = {
        ConditioningMechanism.SUM: jnp.ones((self.batch_size, self.cond_dim)),
    }
    self.adaptive_norm_emb = {
        ConditioningMechanism.ADAPTIVE_NORM: jnp.ones(
            (self.batch_size, self.cond_dim)
        ),
    }
    self.mlp_module = mlp.ConditionalMLP(
        hidden_sizes_preprocess=(32, 16),
        hidden_sizes_postprocess=(32, 16),
        activation='relu',
        dropout_rate=0.0,
        zero_init_output=False,
        conditioning_mechanism=ConditioningMechanism.CONCATENATE,
    )
    self.unet_module = unet.Unet(
        base_channels=8,
        channels_multiplier=(2,),
        num_residual_blocks=(2,),
        downsample_method=arch_typing.DownsampleType.AVG_POOL,
        upsample_method=arch_typing.UpsampleType.NEAREST,
        dropout_rate=(0.0,),
        bottleneck_dropout_rate=0.0,
        self_attention_bool=(False,),
        cross_attention_bool=(False,),
        attention_num_heads=arch_typing.INVALID_INT,
        attention_head_dim=8,
        attention_normalize_qk=False,
        attention_use_rope=False,
        attention_rope_position_type=arch_typing.RoPEPositionType.SQUARE,
        normalization_type=arch_typing.NormalizationType.RMS_NORM,
        normalization_num_groups=0,
        activation='relu',
        skip_connection_method=arch_typing.SkipConnectionMethod.UNNORMALIZED_ADD,
    )

  # TokenEmbedder tests
  @parameterized.parameters(
      itertools.product(
          # embedding_dim
          [1, 16, 100],
          # process_num_categories
          [1, 16, 256, 1024],
      )
  )
  def test_token_embedder_output_shape(
      self,
      embedding_dim: int,
      process_num_categories: int,
  ):
    """Tests the output shape of the TokenEmbedder."""
    token_embedder = discrete.TokenEmbedder(
        process_num_categories=process_num_categories,
        embedding_dim=embedding_dim,
    )
    variables = token_embedder.init(
        self.key, self.discrete_x, is_training=self.is_training
    )
    output = token_embedder.apply(
        variables, self.discrete_x, is_training=self.is_training
    )
    data_shape = self.shape[:-1]
    self.assertEqual(
        (self.batch_size, *data_shape, embedding_dim), output.shape
    )
    self.assertEqual(output.dtype, jnp.float32)
    expected_shape = (process_num_categories, embedding_dim)
    self.assertEqual(
        expected_shape,
        variables['params']['Token_Embedding']['embedding'].shape,
    )

  # ConditionalDiscreteBackbone tests
  @parameterized.parameters(
      itertools.product(
          # embedding_dim
          [1, 16, 100],
          # process_num_categories
          [1, 16, 256, 1024],
          # num_categories
          [1, 16, 250, 1023],
      )
  )
  def test_conditional_discrete_backbone_with_unet_output_shape(
      self,
      embedding_dim: int,
      process_num_categories: int,
      num_categories: int,
  ):
    """Tests the output shape of the MLP."""
    token_embedder = discrete.TokenEmbedder(
        process_num_categories=process_num_categories,
        embedding_dim=embedding_dim,
        adapt_to_image_like_data=True,
    )
    token_projector = discrete.DenseTokenProjector(
        num_categories=num_categories,
        embedding_dim=embedding_dim,
        adapt_to_image_like_data=True,
    )
    discrete_module = discrete.ConditionalDiscreteBackbone(
        base_backbone=self.unet_module,
        token_embedder=token_embedder,
        token_projector=token_projector,
    )

    variables = discrete_module.init(
        self.key,
        self.discrete_x,
        conditioning_embeddings=self.adaptive_norm_emb,
        is_training=self.is_training,
    )
    output = discrete_module.apply(
        variables,
        self.discrete_x,
        conditioning_embeddings=self.adaptive_norm_emb,
        is_training=self.is_training,
    )
    data_shape = self.shape[:-1]
    self.assertEqual(
        (self.batch_size, *data_shape, num_categories), output.shape
    )
    self.assertEqual(output.dtype, jnp.float32)

  @parameterized.parameters(
      itertools.product(
          # embedding_dim
          [1, 16, 100],
          # process_num_categories
          [1, 16, 256, 1024],
          # num_categories
          [1, 16, 250, 1023],
          # adapt_to_image_like_data
          [True, False],
      )
  )
  def test_conditional_discrete_backbone_with_mlp_output_shape(
      self,
      embedding_dim: int,
      process_num_categories: int,
      num_categories: int,
      adapt_to_image_like_data: bool,
  ):
    """Tests the output shape of the MLP."""
    token_embedder = discrete.TokenEmbedder(
        process_num_categories=process_num_categories,
        embedding_dim=embedding_dim,
        adapt_to_image_like_data=adapt_to_image_like_data,
    )

    token_projector = discrete.DenseTokenProjector(
        num_categories=num_categories,
        embedding_dim=embedding_dim,
        adapt_to_image_like_data=adapt_to_image_like_data,
    )

    discrete_module = discrete.ConditionalDiscreteBackbone(
        base_backbone=self.mlp_module,
        token_embedder=token_embedder,
        token_projector=token_projector,
    )

    variables = discrete_module.init(
        self.key,
        self.discrete_x,
        conditioning_embeddings=self.concatenate_emb,
        is_training=self.is_training,
    )
    output = discrete_module.apply(
        variables,
        self.discrete_x,
        conditioning_embeddings=self.concatenate_emb,
        is_training=self.is_training,
    )
    data_shape = self.shape[:-1]
    self.assertEqual(
        (self.batch_size, *data_shape, num_categories), output.shape
    )
    self.assertEqual(output.dtype, jnp.float32)

  def test_failure_if_mismatch_embedding_dims(self):
    """Tests the output shape of the MLP."""
    embedder_embedding_dim = 16
    projector_embedding_dim = 32

    process_num_categories = 16
    num_categories = 16
    adapt_to_image_like_data = True

    token_embedder = discrete.TokenEmbedder(
        process_num_categories=process_num_categories,
        embedding_dim=embedder_embedding_dim,
        adapt_to_image_like_data=adapt_to_image_like_data,
    )

    token_projector = discrete.DenseTokenProjector(
        num_categories=num_categories,
        embedding_dim=projector_embedding_dim,
        adapt_to_image_like_data=adapt_to_image_like_data,
    )

    with self.assertRaisesRegex(
        ValueError, 'The embedding dimension of the token embedder.*'
    ):
      discrete.ConditionalDiscreteBackbone(
          base_backbone=self.mlp_module,
          token_embedder=token_embedder,
          token_projector=token_projector,
      )


if __name__ == '__main__':
  absltest.main()
