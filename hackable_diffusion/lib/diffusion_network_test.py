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

"""Tests for diffusion_network and its components."""

from collections.abc import Mapping

import chex
from flax import linen as nn
from hackable_diffusion.lib import diffusion_network
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import multimodal
from hackable_diffusion.lib import test_helpers
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import conditioning_encoder
from hackable_diffusion.lib.architecture import discrete
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import unet
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.inference import diffusion_inference
from hackable_diffusion.lib.inference import guidance
from hackable_diffusion.lib.inference import wrappers
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
TimeRescaler = diffusion_network.TimeRescaler
InputRescaler = diffusion_network.InputRescaler
GaussianProcess = gaussian.GaussianProcess
IdentityBackbone = test_helpers.IdentityBackbone

################################################################################
# MARK: Test and Configs
################################################################################


UNET_CONFIG = {
    'base_channels': 16,
    'channels_multiplier': (1, 2),
    'self_attention_bool': (False, True),
    'cross_attention_bool': (True, True),
    'dropout_rate': (0.0, 0.0),
    'bottleneck_dropout_rate': 0.0,
    'num_residual_blocks': (1, 1),
    'zero_init_output': False,
    'attention_num_heads': 2,
    'attention_head_dim': arch_typing.INVALID_INT,
    'attention_normalize_qk': True,
    'attention_use_rope': False,
    'attention_rope_position_type': arch_typing.RoPEPositionType.SQUARE,
    'normalization_type': normalization.NormalizationType.GROUP_NORM,
    'normalization_num_groups': 4,
    'skip_connection_method': arch_typing.SkipConnectionMethod.UNNORMALIZED_ADD,
    'downsample_method': arch_typing.DownsampleType.MAX_POOL,
    'upsample_method': arch_typing.UpsampleType.NEAREST,
    'activation': 'silu',
}

LOGSNR_RESCALER = diffusion_network.LogSnrTimeRescaler(
    schedule=schedules.RFSchedule()
)

MAGNITUDE_INPUT_RESCALER = diffusion_network.MagnitudeScheduleInputRescaler(
    schedule=schedules.RFSchedule()
)


################################################################################
# MARK: DiffusionNetwork Tests
################################################################################


class DiffusionNetworkTest(parameterized.TestCase):
  """Tests for the DiffusionNetwork and its components."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.img_shape = (self.batch_size, 32, 32, 3)
    self.discrete_img_shape = (self.batch_size, 32, 32, 3, 1)
    self.is_training = True

    self.t = jnp.ones((self.batch_size,))
    self.xt = jnp.ones(self.img_shape)
    self.discrete_xt = jnp.ones(self.discrete_img_shape, dtype=jnp.int32)
    self.conditioning = {
        'label1': jnp.arange(self.batch_size),
        'label2': jnp.arange(self.batch_size),
    }
    self.schedule = schedules.RFSchedule()
    self.process = GaussianProcess(schedule=self.schedule)

    self.time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=16,
        num_features=32,
    )
    self.cond_encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=self.time_encoder,
        conditioning_embedders={
            'label_foo': conditioning_encoder.LabelEmbedder(
                conditioning_key='label1',
                num_classes=10,
                num_features=16,
            ),
            'label_bar': conditioning_encoder.LabelEmbedder(
                conditioning_key='label2',
                num_classes=10,
                num_features=16,
            ),
        },
        embedding_merging_method=arch_typing.EmbeddingMergeMethod.CONCAT,
        conditioning_rules={
            'time': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label_foo': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label_bar': arch_typing.ConditioningMechanism.CROSS_ATTENTION,
        },
    )
    self.backbone = unet.Unet(**UNET_CONFIG)

  # MARK: Time Rescaler Tests

  def test_logsnr_time_rescaler(self):
    rescaler = diffusion_network.LogSnrTimeRescaler(schedule=self.schedule)
    rescaled_time = rescaler(self.t)
    expected_time = self.schedule.logsnr(self.t)
    self.assertTrue(jnp.allclose(rescaled_time, expected_time))

  def test_logsnr_time_rescaler_with_postprocessing(self):
    postprocess_fn = lambda x: x**2
    rescaler = diffusion_network.LogSnrTimeRescaler(
        schedule=self.schedule,
        postprocess_fn=postprocess_fn,
    )
    rescaled_time = rescaler(self.t)
    expected_time = postprocess_fn(self.schedule.logsnr(self.t))
    self.assertTrue(jnp.allclose(rescaled_time, expected_time))

  # MARK: Input Rescaler Tests

  def test_magnitude_schedule_input_rescaler(self):
    rescaler = diffusion_network.MagnitudeScheduleInputRescaler(
        schedule=self.schedule
    )
    rescaled_inputs = rescaler(self.t, self.xt)

    alpha_t = self.schedule.alpha(self.t)
    sigma_t = self.schedule.sigma(self.t)
    magnitude = jnp.sqrt(jnp.square(alpha_t) + jnp.square(sigma_t))
    # Mimic bcast_right
    magnitude = magnitude.reshape((-1,) + (1,) * (self.xt.ndim - 1))
    expected_inputs = self.xt / magnitude

    self.assertTrue(jnp.allclose(rescaled_inputs, expected_inputs))

  # MARK: DiffusionNetwork Tests

  @parameterized.named_parameters(
      ('no_rescaler', None, None),
      ('time_rescaler', LOGSNR_RESCALER, None),
      ('input_rescaler', None, MAGNITUDE_INPUT_RESCALER),
      ('both_rescale', LOGSNR_RESCALER, MAGNITUDE_INPUT_RESCALER),
  )
  def test_diffusion_network_output_shape(
      self,
      time_rescaler: TimeRescaler,
      input_rescaler: InputRescaler,
  ):
    """Tests DiffusionNetwork output shape."""

    network = diffusion_network.DiffusionNetwork(
        backbone_network=self.backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='x0',
        time_rescaler=time_rescaler,
        input_rescaler=input_rescaler,
    )
    variables = network.init(
        self.key, self.t, self.xt, self.conditioning, self.is_training
    )
    output = network.apply(
        variables, self.t, self.xt, self.conditioning, self.is_training
    )
    self.assertIsInstance(output, dict)
    self.assertIn('x0', output)
    self.assertEqual(output['x0'].shape, self.xt.shape)

  @parameterized.named_parameters(
      ('no_mask', False),
      ('with_mask', True),
  )
  def test_discrete_diffusion_network_variables_shapes(
      self, use_masked_process: bool
  ):
    """Tests DiffusionNetwork output shape."""

    vocab_size = 256

    process_num_categories = (
        vocab_size + 1 if use_masked_process else vocab_size
    )
    num_categories = vocab_size

    discrete_backbone = discrete.ConditionalDiscreteBackbone(
        base_backbone=self.backbone,
        token_embedder=discrete.TokenEmbedder(
            process_num_categories=process_num_categories,
            embedding_dim=16,
            adapt_to_image_like_data=True,
        ),
        token_projector=discrete.DenseProjector(
            num_categories=num_categories,
            embedding_dim=16,
            adapt_to_image_like_data=True,
        ),
    )

    layer = diffusion_network.DiffusionNetwork(
        backbone_network=discrete_backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='logits',
        data_dtype=jnp.int32,
    )
    variables = layer.init(
        self.key, self.t, self.discrete_xt, self.conditioning, self.is_training
    )
    output = layer.apply(
        variables, self.t, self.discrete_xt, self.conditioning, self.is_training
    )
    self.assertIsInstance(output, dict)
    self.assertIn('logits', output)
    self.assertEqual(
        self.discrete_xt.shape[:-1] + (vocab_size,), output['logits'].shape
    )


################################################################################
# MARK: SelfConditioningDiffusionNetwork Tests
################################################################################


class SelfConditioningBackbone(nn.Module, arch_typing.ConditionalBackbone):
  """Backbone for self-conditioning tests.

  Accepts input of shape (B, ..., input_channels + num_classes) and returns
  output of shape (B, ..., num_classes).  The backbone simply applies a dense
  layer so the output depends on the input content.
  """

  num_classes: int = 4

  @nn.compact
  def __call__(
      self,
      x: arch_typing.DataTree,
      conditioning_embeddings: Mapping[
          arch_typing.ConditioningMechanism, Float['batch ...']
      ],
      is_training: bool,
  ) -> arch_typing.DataTree:
    return nn.Dense(features=self.num_classes)(x)


class SelfConditioningDiffusionNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    class MockProcess:

      def __init__(self, num_categories):
        self.num_categories = num_categories

      def corrupt(self, key, x0, time):
        raise NotImplementedError()

      def sample_from_invariant(self, key, data_spec):
        raise NotImplementedError()

      def convert_predictions(self, prediction, xt, time):
        raise NotImplementedError()

      def get_schedule_info(self, time):
        raise NotImplementedError()

    self.key = jax.random.PRNGKey(0)
    self.batch_size = 2
    self.input_channels = 1
    self.num_categories = 4
    self.spatial_shape = (8, 8)
    self.xt_shape = (
        self.batch_size, *self.spatial_shape, self.input_channels
    )
    self.t = jnp.ones((self.batch_size,))
    self.xt = jnp.ones(self.xt_shape)
    self.conditioning = {
        'label1': jnp.arange(self.batch_size),
    }
    self.process = MockProcess(self.num_categories)

    self.time_encoder = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=16,
        num_features=32,
    )
    self.cond_encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=self.time_encoder,
        conditioning_embedders={
            'label': conditioning_encoder.LabelEmbedder(
                conditioning_key='label1',
                num_classes=10,
                num_features=16,
            ),
        },
        embedding_merging_method=arch_typing.EmbeddingMergeMethod.CONCAT,
        conditioning_rules={
            'time': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
        },
    )
    self.backbone = SelfConditioningBackbone(
        num_classes=self.num_categories
    )

  def _make_network(
      self, self_cond_prob: float = 0.5
  ) -> diffusion_network.SelfConditioningDiffusionNetwork:
    return diffusion_network.SelfConditioningDiffusionNetwork(
        backbone_network=self.backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='logits',
        data_dtype=jnp.float32,
        process=self.process,  # type: ignore[wrong-arg-types]
        self_cond_prob=self_cond_prob,
    )

  def test_output_shape(self):
    network = self._make_network()
    variables = network.init(
        {'params': self.key, 'self_conditioning': self.key},
        self.t, self.xt, self.conditioning, True,
    )

    output = network.apply(
        variables,
        self.t, self.xt, self.conditioning, True,
        rngs={'self_conditioning': self.key},
    )

    self.assertIsInstance(output, dict)
    self.assertIn('logits', output)

    expected_shape = (
        self.batch_size, *self.spatial_shape, self.num_categories
    )

    self.assertEqual(output['logits'].shape, expected_shape)

  def test_self_cond_prob_zero_skips_self_cond(self):
    network = self._make_network(self_cond_prob=0.0)
    variables = network.init(
        {'params': self.key, 'self_conditioning': self.key},
        self.t, self.xt, self.conditioning, True,
    )

    output_no_sc = network.apply(
        variables,
        self.t, self.xt, self.conditioning, True,
        rngs={'self_conditioning': self.key},
    )
    # With prob=1.0, self-conditioning is always applied (different output).
    network_always = self._make_network(self_cond_prob=1.0)
    output_always = network_always.apply(
        variables,
        self.t,
        self.xt,
        self.conditioning,
        True,
        rngs={'self_conditioning': self.key},
    )

    self.assertFalse(
        jnp.allclose(output_no_sc['logits'], output_always['logits']),
        msg='Outputs should differ since self-conditioning changes the input.',
    )

  def test_self_cond_prob_one_always_self_conditions(self):
    network = self._make_network(self_cond_prob=1.0)
    variables = network.init(
        {'params': self.key, 'self_conditioning': self.key},
        self.t, self.xt, self.conditioning, True,
    )
    # Run twice with different RNG — should give the same result since
    # self_cond_prob=1.0 means the random draw has no effect.
    output_a = network.apply(
        variables,
        self.t, self.xt, self.conditioning, True,
        rngs={'self_conditioning': jax.random.PRNGKey(42)},
    )
    output_b = network.apply(
        variables,
        self.t, self.xt, self.conditioning, True,
        rngs={'self_conditioning': jax.random.PRNGKey(99)},
    )

    chex.assert_trees_all_close(output_a, output_b)

  def test_inference_always_self_conditions(self):
    # Even with self_cond_prob=0.0, inference should self-condition.
    network = self._make_network(self_cond_prob=0.0)
    variables = network.init(
        {'params': self.key, 'self_conditioning': self.key},
        self.t, self.xt, self.conditioning, True,
    )
    # Inference output (is_training=False).
    output_infer = network.apply(
        variables,
        self.t, self.xt, self.conditioning, False,
    )
    # Training with self_cond_prob=1.0 should match inference.
    network_always = self._make_network(self_cond_prob=1.0)

    output_train_sc = network_always.apply(
        variables,
        self.t, self.xt, self.conditioning, True,
        rngs={'self_conditioning': self.key},
    )

    chex.assert_trees_all_close(output_infer, output_train_sc)

  def test_element_wise_self_conditioning(self):
    batch_size = 100
    xt_shape = (batch_size, *self.spatial_shape, self.input_channels)
    xt = jnp.ones(xt_shape)
    t = jnp.ones((batch_size,))
    conditioning = {'label1': jnp.zeros((batch_size,), dtype=jnp.int32)}

    network_zero = self._make_network(self_cond_prob=0.0)
    network_one = self._make_network(self_cond_prob=1.0)
    network_half = self._make_network(self_cond_prob=0.5)

    variables = network_zero.init(
        {'params': self.key, 'self_conditioning': self.key},
        t,
        xt,
        conditioning,
        True,
    )

    output_zero = network_zero.apply(
        variables,
        t,
        xt,
        conditioning,
        True,
        rngs={'self_conditioning': self.key},
    )
    output_one = network_one.apply(
        variables,
        t,
        xt,
        conditioning,
        True,
        rngs={'self_conditioning': self.key},
    )
    output_half = network_half.apply(
        variables,
        t,
        xt,
        conditioning,
        True,
        rngs={'self_conditioning': self.key},
    )

    logits_zero = output_zero['logits']
    logits_one = output_one['logits']
    logits_half = output_half['logits']

    matches_zero = jnp.all(
        jnp.isclose(logits_half, logits_zero), axis=(1, 2, 3)
    )
    matches_one = jnp.all(jnp.isclose(logits_half, logits_one), axis=(1, 2, 3))

    # Every element must match either zero or one
    self.assertTrue(jnp.all(matches_zero | matches_one))

    # With a fixed seed and batch_size=100, the mask is deterministic and
    # contains a mix of True/False values.
    self.assertFalse(jnp.all(matches_zero))
    self.assertFalse(jnp.all(matches_one))

  def test_default_self_cond_prob(self):
    network = diffusion_network.SelfConditioningDiffusionNetwork(
        backbone_network=self.backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='logits',
        process=self.process,  # type: ignore[wrong-arg-types]
    )

    self.assertEqual(network.self_cond_prob, 0.5)

  def test_invalid_prediction_type_raises(self):
    with self.assertRaisesRegex(ValueError, 'prediction_type'):
      diffusion_network.SelfConditioningDiffusionNetwork(
          backbone_network=self.backbone,
          conditioning_encoder=self.cond_encoder,
          prediction_type='x0',
          process=self.process,  # type: ignore[wrong-arg-types]
      )


################################################################################
# MARK: MultiModalDiffusionNetwork Tests
################################################################################


class MultiModalDiffusionNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.img_shape = (self.batch_size, 32, 32, 3)
    self.is_training = True

  @parameterized.named_parameters(
      ('dict', 'dict'),
      ('list', 'list'),
      ('tuple', 'tuple'),
  )
  def test_multimodal_diffusion_network(self, input_type: str):
    time_encoder_1 = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=16,
        num_features=32,
    )
    time_encoder_2 = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=16,
        num_features=32,
    )

    if input_type == 'dict':
      t = {
          'data_1': jnp.ones((self.batch_size,)),
          'data_2': {'data_3': jnp.ones((self.batch_size,))},
      }
      xt = {
          'data_1': jnp.ones(self.img_shape),
          'data_2': {'data_3': jnp.ones(self.img_shape)},
      }
      conditioning = {
          'label1': jnp.arange(self.batch_size),
          'label2': jnp.arange(self.batch_size),
      }
      time_embedders = {
          'data_1': time_encoder_1,
          'data_2': {'data_3': time_encoder_2},
      }
      prediction_type = {'data_1': 'x0', 'data_2': {'data_3': 'velocity'}}
      data_dtype = {'data_1': jnp.float32, 'data_2': {'data_3': jnp.float32}}
    elif input_type == 'list':
      t = [jnp.ones((self.batch_size,)), jnp.ones((self.batch_size,))]
      xt = [jnp.ones(self.img_shape), jnp.ones(self.img_shape)]
      conditioning = {
          'label1': jnp.arange(self.batch_size),
          'label2': jnp.arange(self.batch_size),
      }
      time_embedders = [time_encoder_1, time_encoder_2]
      prediction_type = ['x0', 'velocity']
      data_dtype = [jnp.float32, jnp.float32]
    elif input_type == 'tuple':
      t = (jnp.ones((self.batch_size,)), jnp.ones((self.batch_size,)))
      xt = (jnp.ones(self.img_shape), jnp.ones(self.img_shape))
      conditioning = {
          'label1': jnp.arange(self.batch_size),
          'label2': jnp.arange(self.batch_size),
      }
      time_embedders = (time_encoder_1, time_encoder_2)
      prediction_type = ('x0', 'velocity')
      data_dtype = (jnp.float32, jnp.float32)
    else:
      raise ValueError(f'Unknown input type {input_type}')

    time_encoder = multimodal.NestedTimeEmbedder(time_embedders=time_embedders)
    cond_encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders={
            'label_foo': conditioning_encoder.LabelEmbedder(
                conditioning_key='label1',
                num_classes=10,
                num_features=16,
            ),
            'label_bar': conditioning_encoder.LabelEmbedder(
                conditioning_key='label2',
                num_classes=10,
                num_features=16,
            ),
        },
        embedding_merging_method=arch_typing.EmbeddingMergeMethod.CONCAT,
        conditioning_rules={
            'time': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label_foo': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label_bar': arch_typing.ConditioningMechanism.CROSS_ATTENTION,
        },
    )

    backbone = IdentityBackbone()

    network = diffusion_network.MultiModalDiffusionNetwork(
        backbone_network=backbone,
        conditioning_encoder=cond_encoder,
        prediction_type=prediction_type,
        data_dtype=data_dtype,
    )

    variables = network.init(
        self.key,
        time=t,
        xt=xt,
        conditioning=conditioning,
        is_training=self.is_training,
    )
    output = network.apply(
        variables,
        time=t,
        xt=xt,
        conditioning=conditioning,
        is_training=self.is_training,
    )

    modified_t = jax.tree.map(
        lambda t, prediction_type: {prediction_type: t}, t, prediction_type
    )

    chex.assert_trees_all_equal_structs(modified_t, output)


################################################################################
# MARK: NestedDiffusionInference Tests
################################################################################


class NestedDiffusionInferenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.img_shape = (self.batch_size, 32, 32, 3)
    self.is_training = True
    self.schedule = schedules.RFSchedule()

    time_encoder_1 = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=16,
        num_features=32,
    )
    time_encoder_2 = conditioning_encoder.SinusoidalTimeEmbedder(
        activation='silu',
        embedding_dim=16,
        num_features=32,
    )
    time_embedders = {
        'data_1': time_encoder_1,
        'data_2': {'data_3': time_encoder_2},
    }
    time_encoder = multimodal.NestedTimeEmbedder(time_embedders=time_embedders)
    self.nested_cond_encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_encoder,
        conditioning_embedders={
            'label_foo': conditioning_encoder.LabelEmbedder(
                conditioning_key='label_foo',
                num_classes=10,
                num_features=16,
            ),
            'label_bar': conditioning_encoder.LabelEmbedder(
                conditioning_key='label_bar',
                num_classes=10,
                num_features=16,
            ),
        },
        embedding_merging_method=arch_typing.EmbeddingMergeMethod.CONCAT,
        conditioning_rules={
            'time': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label_foo': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
            'label_bar': arch_typing.ConditioningMechanism.CROSS_ATTENTION,
        },
    )

    self.nested_t = {
        'data_1': jnp.ones((self.batch_size,)),
        'data_2': {'data_3': jnp.ones((self.batch_size,))},
    }
    self.nested_xt = {
        'data_1': jnp.ones(self.img_shape),
        'data_2': {'data_3': jnp.ones(self.img_shape)},
    }
    self.nested_conditioning = {
        'label_foo': jnp.arange(self.batch_size),
        'label_bar': jnp.arange(self.batch_size),
    }
    self.nested_prediction_type = {
        'data_1': 'x0',
        'data_2': {'data_3': 'velocity'},
    }
    self.nested_data_dtype = {
        'data_1': jnp.float32,
        'data_2': {'data_3': jnp.float32},
    }
    self.nested_backbone = IdentityBackbone()

    self.nested_guidance_fn = multimodal.NestedGuidanceFn(
        guidance_fns={
            'data_1': guidance.ScalarGuidanceFn(guidance=0.0),
            'data_2': {'data_3': guidance.ScalarGuidanceFn(guidance=1.0)},
        }
    )

  def test_nested_inference(self):
    layer = diffusion_network.MultiModalDiffusionNetwork(
        backbone_network=self.nested_backbone,
        conditioning_encoder=self.nested_cond_encoder,
        prediction_type=self.nested_prediction_type,
        data_dtype=self.nested_data_dtype,
    )
    variables = layer.init(
        self.key,
        time=self.nested_t,
        xt=self.nested_xt,
        conditioning=self.nested_conditioning,
        is_training=self.is_training,
    )
    params = variables['params']
    shifted_params = jax.tree.map(lambda x: x + 1e-4, params)
    base_inference_fn = wrappers.FlaxLinenInferenceFn(
        network=layer, params=shifted_params
    )
    inference_fn = diffusion_inference.GuidedDiffusionInferenceFn(
        base_inference_fn=base_inference_fn,
        guidance_fn=self.nested_guidance_fn,
    )
    output = inference_fn(
        xt=self.nested_xt,
        conditioning=self.nested_conditioning,
        time=self.nested_t,
    )
    modified_nested_t = jax.tree.map(
        lambda t, prediction_type: {prediction_type: t},
        self.nested_t,
        self.nested_prediction_type,
    )

    chex.assert_trees_all_equal_structs(output, modified_nested_t)


if __name__ == '__main__':
  absltest.main()
