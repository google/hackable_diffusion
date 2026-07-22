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

"""Tests for the conditioning encoder."""

from hackable_diffusion.lib.architecture import conditioning_encoder
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

SumEmbeddings = conditioning_encoder.SumEmbeddings
ConcatEmbeddings = conditioning_encoder.ConcatEmbeddings


################################################################################
# MARK: Tests
################################################################################


class EncodeConditioningTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 4
    self.num_features = 32
    self.num_classes = 10
    self.embedding_dim = 5
    self.conditioning_dropout_rate = 0.1
    self.rng = jax.random.PRNGKey(0)

  @parameterized.named_parameters(
      (
          'test1',
          SumEmbeddings(),
          'adaptive_norm',
          True,
      ),
      (
          'test2',
          ConcatEmbeddings(),
          'cross_attention',
          False,
      ),
  )
  def test_basic(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    label_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=self.num_classes, num_features=self.num_features
    )
    conditioning_encoders = {'label': label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size)}
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    if isinstance(merge_embeddings_fn, SumEmbeddings):
      expected_shape = (self.batch_size, self.num_features)
    elif isinstance(merge_embeddings_fn, ConcatEmbeddings):
      expected_shape = (self.batch_size, 2 * self.num_features)
    else:
      raise ValueError(f'Unknown method {merge_embeddings_fn}')

    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          SumEmbeddings(),
          'adaptive_norm',
          True,
      ),
      (
          'test2',
          ConcatEmbeddings(),
          'cross_attention',
          False,
      ),
  )
  def test_mlp_embedder(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    label_encoder = conditioning_encoder.MLPEmbedder(
        num_features=self.num_features,
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
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size, dtype=jnp.float32)}
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    if isinstance(merge_embeddings_fn, SumEmbeddings):
      expected_shape = (self.batch_size, self.num_features)
    elif isinstance(merge_embeddings_fn, ConcatEmbeddings):
      expected_shape = (self.batch_size, 2 * self.num_features)
    else:
      raise ValueError(f'Unknown method {merge_embeddings_fn}')

    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          SumEmbeddings(),
          'adaptive_norm',
          True,
      ),
      (
          'test2',
          ConcatEmbeddings(),
          'cross_attention',
          False,
      ),
  )
  def test_mlp_embedder_process_multiple_keys(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    multi_label_encoder = conditioning_encoder.MLPEmbedder(
        num_features=self.num_features,
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
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = {
        'label1': jnp.arange(self.batch_size, dtype=jnp.float32),
        'label2': jnp.arange(self.batch_size, dtype=jnp.float32) + 1,
    }
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    if isinstance(merge_embeddings_fn, SumEmbeddings):
      expected_shape = (self.batch_size, self.num_features)
    elif isinstance(merge_embeddings_fn, ConcatEmbeddings):
      expected_shape = (self.batch_size, 2 * self.num_features)
    else:
      raise ValueError(f'Unknown method {merge_embeddings_fn}')

    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          SumEmbeddings(),
          'adaptive_norm',
          True,
      ),
      (
          'test2',
          ConcatEmbeddings(),
          'cross_attention',
          False,
      ),
  )
  def test_mlp_embedder_fails_on_missing_key(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      is_training,
  ):
    """Tests basic functionality with different merging and conditioning."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    label_encoder = conditioning_encoder.MLPEmbedder(
        num_features=self.num_features,
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
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size, dtype=jnp.float32)}
    with self.assertRaises(
        ValueError,
        msg=(
            'Conditioning key missing_key not found in conditioning. Available'
            " keys: ['label']"
        ),
    ):
      _ = encoder.init(self.rng, t, c, is_training=is_training)['params']

  def test_field_selector_embedder(self):
    """Tests FieldSelector with CROSS_ATTENTION."""
    image_shape = (64, 64, 3)
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    image_selector = conditioning_encoder.FieldSelector(
        field_name='image',
        data_spec=image_shape,
    )
    conditioning_encoders = {'image': image_selector}
    conditioning_rules = {
        'time': 'adaptive_norm',
        'image': 'cross_attention',
    }
    merge_embeddings_fn = ConcatEmbeddings()

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=0.0,
    )

    t = jnp.ones((self.batch_size,))
    c = {'image': jnp.ones((self.batch_size,) + image_shape)}
    params = encoder.init(self.rng, t, c, is_training=False)['params']

    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])
    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=False,
        rngs={'dropout': self.rng},
    )

    self.assertIn('cross_attention', output)
    self.assertEqual(
        output['cross_attention'].shape,
        (self.batch_size,) + image_shape,
    )
    self.assertTrue(
        jnp.all(output['cross_attention'] == c['image'])
    )

    self.assertIn('adaptive_norm', output)
    self.assertEqual(
        output['adaptive_norm'].shape,
        (self.batch_size, self.num_features),
    )

  def test_field_selector_embedder_fails_on_missing_key(self):
    """Tests FieldSelector raises ValueError on missing key."""
    image_shape = (64, 64, 3)
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    image_selector = conditioning_encoder.FieldSelector(
        field_name='image',
        data_spec=image_shape,
    )
    conditioning_encoders = {'image': image_selector}
    conditioning_rules = {
        'time': 'adaptive_norm',
        'image': 'cross_attention',
    }
    merge_embeddings_fn = ConcatEmbeddings()

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=0.0,
    )

    t = jnp.ones((self.batch_size,))
    c = {'wrong_key': jnp.ones((self.batch_size,) + image_shape)}
    with self.assertRaisesRegex(
        ValueError,
        'Conditioning key image not found in conditioning. Available keys:'
        " \\['wrong_key'\\]",
    ):
      encoder.init(self.rng, t, c, is_training=False)

  @parameterized.named_parameters(
      (
          'test1',
          ConcatEmbeddings(),
          'cross_attention',
          8,
          16,
          False,
      ),
  )
  def test_different_num_features(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      time_encode_num_features,
      label_encode_num_features,
      is_training,
  ):
    """Tests encoders with different feature dims when concatenation is used."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=time_encode_num_features,
    )
    label_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=self.num_classes, num_features=label_encode_num_features
    )
    conditioning_encoders = {'label': label_encoder}
    conditioning_rules = {
        'time': conditioning_mechanism,
        'label': conditioning_mechanism,
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size)}
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    expected_shape = (
        self.batch_size,
        time_encode_num_features + label_encode_num_features,
    )
    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          ConcatEmbeddings(),
          'cross_attention',
          8,
          9,
          10,
          False,
      ),
      (
          'test2',
          ConcatEmbeddings(),
          'cross_attention',
          8,
          9,
          10,
          True,
      ),
  )
  def test_multilabel(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      time_encode_num_features,
      label1_encode_num_features,
      label2_encode_num_features,
      is_training,
  ):
    """Tests the unconditional case where one of the conditionings is None."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=time_encode_num_features,
    )
    label1_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=self.num_classes,
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
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = {
        'label1': jnp.arange(self.batch_size),
        'label2': jnp.arange(self.batch_size) + 1,
    }
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    expected_shape = (
        self.batch_size,
        time_encode_num_features
        + label1_encode_num_features
        + label2_encode_num_features,
    )
    self.assertEqual(conditional_embedding.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'test1',
          ConcatEmbeddings(),
          'cross_attention',
          8,
          9,
          10,
          False,
      ),
      (
          'test2',
          ConcatEmbeddings(),
          'cross_attention',
          8,
          9,
          10,
          True,
      ),
  )
  def test_unconditional(
      self,
      merge_embeddings_fn,
      conditioning_mechanism,
      time_encode_num_features,
      label1_encode_num_features,
      label2_encode_num_features,
      is_training,
  ):
    """Tests the unconditional case where one of the conditionings is None."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=time_encode_num_features,
    )
    label1_encoder = conditioning_encoder.LabelEmbedder(
        num_classes=self.num_classes, num_features=label1_encode_num_features
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
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=merge_embeddings_fn,
        conditioning_rules=conditioning_rules,
        conditioning_dropout_rate=self.conditioning_dropout_rate,
    )

    t = jnp.ones((self.batch_size,))
    c = None
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']

    # Jit the apply function
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    output = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

    self.assertIn(conditioning_mechanism, output)
    conditional_embedding = output[conditioning_mechanism]

    expected_shape = (
        self.batch_size,
        time_encode_num_features
        + label1_encode_num_features
        + label2_encode_num_features,
    )
    self.assertEqual(conditional_embedding.shape, expected_shape)

  def test_dropout(self):
    """Tests that dropout is correctly applied based on `is_training`."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    conditioning_encoders = {
        'label': conditioning_encoder.LabelEmbedder(
            num_classes=self.num_classes, num_features=self.num_features
        )
    }

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=SumEmbeddings(),
        conditioning_rules={
            'time': 'adaptive_norm',
            'label': 'adaptive_norm',
        },
        conditioning_dropout_rate=1.0,  # Drop all conditioning
    )

    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size)}
    params = encoder.init(self.rng, t, c, is_training=True)['params']
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])

    # With is_training=True, the label embedding should be all zeros.
    output_train = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=True,
        rngs={'dropout': self.rng},
    )
    time_embedding_train = time_encoder.apply(
        {'params': params['time_embedder']}, t
    )
    self.assertTrue(
        jnp.all(output_train['adaptive_norm'] == time_embedding_train)
    )

    # With is_training=False, the label embedding should not be dropped.
    output_eval = jitted_apply(
        {'params': params},
        t,
        c,
        is_training=False,
        rngs={'dropout': self.rng},
    )
    self.assertFalse(
        jnp.all(output_eval['adaptive_norm'] == time_embedding_train)
    )

  def _make_mask_test_encoder(self, conditioning_dropout_rate=0.5):
    """Creates a ConditioningEncoder for mask tests."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    conditioning_encoders = {
        'label': conditioning_encoder.LabelEmbedder(
            num_classes=self.num_classes,
            num_features=self.num_features,
        )
    }
    return conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=SumEmbeddings(),
        conditioning_rules={
            'time': 'adaptive_norm',
            'label': 'adaptive_norm',
        },
        conditioning_dropout_rate=conditioning_dropout_rate,
    )

  def _init_and_apply_mask_test(self, encoder, is_training):
    """Inits params and applies the encoder for mask tests."""
    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size)}
    params = encoder.init(self.rng, t, c, is_training=is_training)['params']
    jitted_apply = jax.jit(encoder.apply, static_argnames=['is_training'])
    return jitted_apply(
        {'params': params},
        t,
        c,
        is_training=is_training,
        rngs={'dropout': self.rng},
    )

  def test_conditioning_mask_present_in_output(self):
    """Tests that `conditioning_mask` is present in the output."""
    encoder = self._make_mask_test_encoder(conditioning_dropout_rate=0.5)
    output = self._init_and_apply_mask_test(encoder, is_training=True)
    self.assertIn('conditioning_mask', output)
    self.assertEqual(output['conditioning_mask'].shape, (self.batch_size,))

  def test_conditioning_mask_all_ones_at_eval(self):
    """Tests that mask is all ones when is_training=False."""
    encoder = self._make_mask_test_encoder(conditioning_dropout_rate=0.5)
    # At eval time, mask should be all ones regardless of dropout rate.
    output = self._init_and_apply_mask_test(encoder, is_training=False)
    self.assertTrue(jnp.all(output['conditioning_mask']))

  def test_conditioning_mask_all_zeros_with_full_dropout(self):
    """Tests that mask is all zeros when dropout_rate=1.0 and is_training."""
    encoder = self._make_mask_test_encoder(conditioning_dropout_rate=1.0)
    # With dropout_rate=1.0 and is_training=True, mask should be all zeros.
    output = self._init_and_apply_mask_test(encoder, is_training=True)
    self.assertFalse(jnp.any(output['conditioning_mask']))

  def test_conditioning_mask_all_ones_with_zero_dropout(self):
    """Tests that mask is all ones when dropout_rate=0.0 and is_training."""
    encoder = self._make_mask_test_encoder(conditioning_dropout_rate=0.0)
    # With dropout_rate=0.0, mask should be all ones even at training time.
    output = self._init_and_apply_mask_test(encoder, is_training=True)
    self.assertTrue(jnp.all(output['conditioning_mask']))

  def test_conditioning_mask_reserved_key_raises(self):
    """Tests that using 'conditioning_mask' in conditioning_rules raises."""
    time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=self.embedding_dim,
        num_features=self.num_features,
    )
    conditioning_encoders = {
        'label': conditioning_encoder.LabelEmbedder(
            num_classes=self.num_classes,
            num_features=self.num_features,
        )
    }
    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders=conditioning_encoders,  # pyrefly: ignore[bad-argument-type]
        merge_embeddings_fn=SumEmbeddings(),
        conditioning_rules={
            'time': 'adaptive_norm',
            'label': 'adaptive_norm',
            'conditioning_mask': 'adaptive_norm',
        },
        conditioning_dropout_rate=0.0,
    )

    t = jnp.ones((self.batch_size,))
    c = {'label': jnp.arange(self.batch_size)}
    with self.assertRaisesRegex(
        ValueError,
        "'conditioning_mask' is a reserved key",
    ):
      encoder.init(self.rng, t, c, is_training=False)


if __name__ == '__main__':
  absltest.main()
