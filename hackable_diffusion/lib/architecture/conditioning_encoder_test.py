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

"""Tests for the conditioning encoder."""

from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import conditioning_encoder
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

EmbeddingMergeMethod = arch_typing.EmbeddingMergeMethod
ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: Tests
################################################################################


class EncodeConditioningTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.SUM,
          ConditioningMechanism.ADAPTIVE_NORM,
          True,
      ),
      (
          'test2',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          False,
      ),
  )
  def test_basic(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    batch_size = 4
    num_features = 32
    num_classes = 10
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    label_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=num_classes, num_features=num_features
    )
    conditioning_encoders = {'label': label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = {'label': jnp.arange(batch_size)}
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params}, t, c, is_training=is_training, rngs={'dropout': rng}
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    if embedding_merging_method == EmbeddingMergeMethod.SUM:
      expected_shape = (batch_size, num_features)
    elif embedding_merging_method == EmbeddingMergeMethod.CONCAT:
      expected_shape = (batch_size, 2 * num_features)
    else:
      raise ValueError(f'Unknown method {embedding_merging_method}')

    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      ('dict', 'dict'),
      ('tuple', 'tuple'),
      ('list', 'list'),
  )
  def test_nested_time_embedder_with_dict(self, input_type: str):
    num_features = 32
    batch_size = 4

    time_encoder_1 = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    time_encoder_2 = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    if input_type == 'dict':
      time_encoder = conditioning_encoder.NestedTimeEmbedder(
          time_embedders={'data_1': time_encoder_1, 'data_2': time_encoder_2},
      )
      time = {
          'data_1': jnp.ones((batch_size,)),
          'data_2': jnp.ones((batch_size,)),
      }
    elif input_type == 'tuple':
      time_encoder = conditioning_encoder.NestedTimeEmbedder(
          time_embedders=(time_encoder_1, time_encoder_2),
      )
      time = (jnp.ones((batch_size,)), jnp.ones((batch_size,)))
    elif input_type == 'list':
      time_encoder = conditioning_encoder.NestedTimeEmbedder(
          time_embedders=[time_encoder_1, time_encoder_2],
      )
      time = [jnp.ones((batch_size,)), jnp.ones((batch_size,))]
    else:
      raise ValueError(f'Unknown input type {input_type}')

    rng = jax.random.PRNGKey(0)
    params = time_encoder.init(rng, time)['params']

    # Jit the apply function
    jitted_apply = jax.jit(time_encoder.apply, static_argnames=['is_training'])

    output = jitted_apply({'params': params}, time, rngs={'dropout': rng})

    self.assertEqual(output.shape, (batch_size, num_features))

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.SUM,
          ConditioningMechanism.ADAPTIVE_NORM,
          True,
      ),
      (
          'test2',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          False,
      ),
  )
  def test_mlp_embedder(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    batch_size = 4
    num_features = 32
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    label_encoder = conditioning_encoder.MLPEmbedder(
        num_features=num_features,
        hidden_sizes=[16, 8],
        conditioning_keys=['label'],
    )
    conditioning_encoders = {'label': label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = {'label': jnp.arange(batch_size, dtype=jnp.float32)}
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params}, t, c, is_training=is_training, rngs={'dropout': rng}
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    if embedding_merging_method == EmbeddingMergeMethod.SUM:
      expected_shape = (batch_size, num_features)
    elif embedding_merging_method == EmbeddingMergeMethod.CONCAT:
      expected_shape = (batch_size, 2 * num_features)
    else:
      raise ValueError(f'Unknown method {embedding_merging_method}')

    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.SUM,
          ConditioningMechanism.ADAPTIVE_NORM,
          True,
      ),
      (
          'test2',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          False,
      ),
  )
  def test_mlp_embedder_process_multiple_keys(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    batch_size = 4
    num_features = 32
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    multi_label_encoder = conditioning_encoder.MLPEmbedder(
        num_features=num_features,
        hidden_sizes=[16, 8],
        conditioning_keys=['label1', 'label2'],
    )
    conditioning_encoders = {'label': multi_label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = {
        'label1': jnp.arange(batch_size, dtype=jnp.float32),
        'label2': jnp.arange(batch_size, dtype=jnp.float32) + 1,
    }
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params}, t, c, is_training=is_training, rngs={'dropout': rng}
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    if embedding_merging_method == EmbeddingMergeMethod.SUM:
      expected_shape = (batch_size, num_features)
    elif embedding_merging_method == EmbeddingMergeMethod.CONCAT:
      expected_shape = (batch_size, 2 * num_features)
    else:
      raise ValueError(f'Unknown method {embedding_merging_method}')

    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.SUM,
          ConditioningMechanism.ADAPTIVE_NORM,
          True,
      ),
      (
          'test2',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          False,
      ),
  )
  def test_mlp_embedder_fails_on_missing_key(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    batch_size = 4
    num_features = 32
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    label_encoder = conditioning_encoder.MLPEmbedder(
        num_features=num_features,
        hidden_sizes=[16, 8],
        conditioning_keys=['missing_key'],
    )
    conditioning_encoders = {'label': label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = {'label': jnp.arange(batch_size, dtype=jnp.float32)}
    rng = jax.random.PRNGKey(0)
    with self.assertRaises(
        ValueError,
        msg=(
            'Conditioning key missing_key not found in conditioning. Available'
            " keys: ['label']"
        ),
    ):
      _ = encoder.init(rng, t, c, is_training=is_training)['params']

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          8,
          16,
          False,
      ),
  )
  def test_different_num_features(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      time_encode_num_features,
      label_encode_num_features,
      is_training,
  ):
    """Tests encoders with different feature dims when concatenation is used."""
    batch_size = 4
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=time_encode_num_features,
    )
    label_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=10, num_features=label_encode_num_features
    )
    conditioning_encoders = {'label': label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = {'label': jnp.arange(batch_size)}
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params}, t, c, is_training=is_training, rngs={'dropout': rng}
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    expected_shape = (
        batch_size,
        time_encode_num_features + label_encode_num_features,
    )
    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          8,
          9,
          10,
          False,
      ),
      (
          'test2',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          8,
          9,
          10,
          True,
      ),
  )
  def test_multilabel(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      time_encode_num_features,
      label1_encode_num_features,
      label2_encode_num_features,
      is_training,
  ):
    """Tests the unconditional case where one of the conditionings is None."""
    batch_size = 4
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=time_encode_num_features,
    )
    label1_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=10,
        num_features=label1_encode_num_features,
        conditioning_key='label1',
    )
    label2_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=8,
        num_features=label2_encode_num_features,
        conditioning_key='label2',
    )

    conditioning_encoders = {
        'label_foo': label1_encoder,
        'label_bar': label2_encoder,
    }
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label_foo': conditioning_mechanism,
        'label_bar': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = {
        'label1': jnp.arange(batch_size),
        'label2': jnp.arange(batch_size) + 1,
    }
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params}, t, c, is_training=is_training, rngs={'dropout': rng}
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    expected_shape = (
        batch_size,
        time_encode_num_features
        + label1_encode_num_features
        + label2_encode_num_features,
    )
    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          8,
          9,
          10,
          False,
      ),
      (
          'test2',
          EmbeddingMergeMethod.CONCAT,
          ConditioningMechanism.CROSS_ATTENTION,
          8,
          9,
          10,
          True,
      ),
  )
  def test_unconditional(
      self,
      embedding_merging_method,
      conditioning_mechanism,
      time_encode_num_features,
      label1_encode_num_features,
      label2_encode_num_features,
      is_training,
  ):
    """Tests the unconditional case where one of the conditionings is None."""
    batch_size = 4
    conditioning_dropout_rate = 0.1
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=time_encode_num_features,
    )
    label1_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=10, num_features=label1_encode_num_features
    )
    label2_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=8, num_features=label2_encode_num_features
    )

    conditioning_encoders = {
        'label1': label1_encoder,
        'label2': label2_encoder,
    }
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label1': conditioning_mechanism,
        'label2': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=embedding_merging_method,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

    t = jnp.ones((batch_size,))
    c = None
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params}, t, c, is_training=is_training, rngs={'dropout': rng}
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    expected_shape = (
        batch_size,
        time_encode_num_features
        + label1_encode_num_features
        + label2_encode_num_features,
    )
    self.assertEqual(conditional_embedding.shape, expected_shape)

  def test_dropout(self):
    """Tests that dropout is correctly applied based on `is_training`."""
    batch_size = 4
    num_features = 32
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=5,
        num_features=num_features,
    )
    conditioning_encoders = {
        'label': conditioning_encoder.LabelEmbedder(
            num_classes=10, num_features=num_features
        )
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,
        embedding_merging_method=EmbeddingMergeMethod.SUM,
        conditioning_rules={
            'time': ConditioningMechanism.ADAPTIVE_NORM,
            'label': ConditioningMechanism.ADAPTIVE_NORM,
        },
        conditioning_dropout_rate=1.0,  # Drop all conditioning
    )

    t = jnp.ones((batch_size,))
    c = {'label': jnp.arange(batch_size)}
    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, t, c, is_training=True)['params']
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    # With is_training=True, the label embedding should be all zeros.
    output_train = jitted_apply(
        {'params': params}, t, c, is_training=True, rngs={'dropout': rng}
    )
    time_embedding_train = time_encoder.apply(
        {'params': params['TimeEmbedder']}, t
    )
    self.assertTrue(
        jnp.all(output_train['adaptive_norm'] == time_embedding_train)
    )

    # With is_training=False, the label embedding should not be dropped.
    output_eval = jitted_apply(
        {'params': params}, t, c, is_training=False, rngs={'dropout': rng}
    )
    self.assertFalse(
        jnp.all(output_eval['adaptive_norm'] == time_embedding_train)
    )


if __name__ == '__main__':
  absltest.main()
