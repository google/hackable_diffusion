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

"""Tests for diffusion_network and its components."""

import chex
from flax import linen as nn
from hackable_diffusion.lib import diffusion_network
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import conditioning_encoder
from hackable_diffusion.lib.architecture import discrete
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import unet
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
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

################################################################################
# MARK: Test Constants, Helper functions and Configs
################################################################################


class IdentityBackbone(arch_typing.ConditionalBackbone):

  @nn.compact
  def __call__(
      self,
      x: arch_typing.DataTree,
      conditioning_embeddings: dict[
          arch_typing.ConditioningMechanism, Float['batch ...']
      ],
      is_training: bool,
  ) -> arch_typing.DataTree:
    return x


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

  # MARK: MultiModalDiffusionNetwork Tests
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

    time_encoder = conditioning_encoder.NestedTimeEmbedder(
        time_embedders=time_embedders
    )
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


if __name__ == '__main__':
  absltest.main()
