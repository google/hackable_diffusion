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

"""Tests for the diffusion_inference module."""

import chex
from flax import linen as nn
from hackable_diffusion.lib import diffusion_network
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import conditioning_encoder
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

################################################################################
# MARK: Constants
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


CONDITIONING_ENCODER = {
    'label_foo': conditioning_encoder.LabelEmbedder(
        num_classes=10,
        num_features=16,
        conditioning_key='label_foo',
    ),
    'label_bar': conditioning_encoder.LabelEmbedder(
        num_classes=10,
        num_features=16,
        conditioning_key='label_bar',
    ),
}
CONDITIONING_RULES = {
    'time': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
    'label_foo': arch_typing.ConditioningMechanism.ADAPTIVE_NORM,
    'label_bar': arch_typing.ConditioningMechanism.CROSS_ATTENTION,
}

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
# MARK: Tests
################################################################################


class DiffusionInferenceTest(parameterized.TestCase):
  """Tests for the DiffusionInferenceFn and its components."""

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.batch_size = 4
    self.img_shape = (self.batch_size, 32, 32, 3)
    self.is_training = True

    self.t = jnp.ones((self.batch_size,)) * 0.5
    self.xt = jnp.ones(self.img_shape)
    self.conditioning = {
        'label_foo': jnp.arange(self.batch_size),
        'label_bar': jnp.arange(self.batch_size),
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
        conditioning_embedders=CONDITIONING_ENCODER,
        embedding_merging_method=arch_typing.EmbeddingMergeMethod.CONCAT,
        conditioning_rules=CONDITIONING_RULES,
    )
    self.backbone = unet.Unet(**UNET_CONFIG)

    # nested inference

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
    time_encoder = conditioning_encoder.NestedTimeEmbedder(
        time_embedders=time_embedders
    )
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

    self.nested_process = {
        'data_1': GaussianProcess(schedule=self.schedule),
        'data_2': {'data_3': GaussianProcess(schedule=self.schedule)},
    }
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

    self.nested_guidance_fn = guidance.NestedGuidanceFn(
        guidance_fns={
            'data_1': guidance.ScalarGuidanceFn(guidance=0.0),
            'data_2': {'data_3': guidance.ScalarGuidanceFn(guidance=1.0)},
        }
    )

  # MARK: Helper Functions Tests

  @parameterized.named_parameters(
      ('no_rescale', None, None),
      ('time_rescale', LOGSNR_RESCALER, None),
      ('input_rescale', None, MAGNITUDE_INPUT_RESCALER),
      ('both_rescale', LOGSNR_RESCALER, MAGNITUDE_INPUT_RESCALER),
  )
  def test_diffusion_inference_fn(
      self,
      time_rescaler: TimeRescaler,
      input_rescaler: InputRescaler,
  ):
    """Tests DiffusionNetwork create_inference_fn."""

    layer = diffusion_network.DiffusionNetwork(
        backbone_network=self.backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='x0',
        time_rescaler=time_rescaler,
        input_rescaler=input_rescaler,
    )
    variables = layer.init(
        self.key,
        time=self.t,
        xt=self.xt,
        conditioning=self.conditioning,
        is_training=self.is_training,
    )
    params = variables['params']
    base_inference_fn = wrappers.FlaxLinenInferenceFn(
        network=layer, params=params
    )
    inference_fn = diffusion_inference.GuidedDiffusionInferenceFn(
        base_inference_fn=base_inference_fn,
    )
    output = inference_fn(
        xt=self.xt, conditioning=self.conditioning, time=self.t
    )

    self.assertIn('x0', output)
    self.assertEqual(output['x0'].shape, self.xt.shape)

  @parameterized.named_parameters(
      ('no_rescale', None, None),
      ('time_rescale', LOGSNR_RESCALER, None),
      ('input_rescale', None, MAGNITUDE_INPUT_RESCALER),
      ('both_rescale', LOGSNR_RESCALER, MAGNITUDE_INPUT_RESCALER),
  )
  def test_unconditional_inference(
      self,
      time_rescaler: TimeRescaler,
      input_rescaler: InputRescaler,
  ):
    """Tests that the unconditional inference function works.

    This corresponds to setting the guidance to -1.0. In that case, the
    inference function should not depend on the value of the conditioning.

    Args:
      time_rescaler: The time rescaler to use for the diffusion model.
      input_rescaler: The input rescaler to use for the diffusion model.
    """

    layer = diffusion_network.DiffusionNetwork(
        backbone_network=self.backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='x0',
        time_rescaler=time_rescaler,
        input_rescaler=input_rescaler,
    )
    variables = layer.init(
        self.key,
        time=self.t,
        xt=self.xt,
        conditioning=self.conditioning,
        is_training=self.is_training,
    )
    params = variables['params']
    shifted_params = jax.tree.map(lambda x: x + 1e-4, params)
    # shift the params by a small amount to ensure that we are not sensitive to
    # zero initialization of the params impacting the conditioning embeddings.

    base_inference_fn = wrappers.FlaxLinenInferenceFn(
        network=layer, params=shifted_params
    )
    inference_fn = diffusion_inference.GuidedDiffusionInferenceFn(
        base_inference_fn=base_inference_fn,
        guidance_fn=guidance.ScalarGuidanceFn(guidance=-1.0),
    )
    other_conditioning = {
        'label_foo': jnp.zeros_like(self.conditioning['label_foo']),
        'label_bar': jnp.zeros_like(self.conditioning['label_bar']),
    }

    output = inference_fn(
        xt=self.xt, conditioning=self.conditioning, time=self.t
    )
    other_output = inference_fn(
        xt=self.xt, conditioning=other_conditioning, time=self.t
    )

    self.assertIn('x0', output)
    self.assertIn('x0', other_output)

    self.assertTrue(jnp.allclose(output['x0'], other_output['x0']))

  @parameterized.named_parameters(
      ('no_rescale', None, None),
      ('time_rescale', LOGSNR_RESCALER, None),
      ('input_rescale', None, MAGNITUDE_INPUT_RESCALER),
      ('both_rescale', LOGSNR_RESCALER, MAGNITUDE_INPUT_RESCALER),
  )
  def test_conditional_inference(
      self,
      time_rescaler: TimeRescaler,
      input_rescaler: InputRescaler,
  ):
    """Tests that the conditional inference function works.

    This corresponds to setting the guidance to a value different than -1.0. In
    that case, the inference function should depend on the value of the
    conditioning.

    Args:
      time_rescaler: The time rescaler to use for the diffusion model.
      input_rescaler: The input rescaler to use for the diffusion model.
    """

    layer = diffusion_network.DiffusionNetwork(
        backbone_network=self.backbone,
        conditioning_encoder=self.cond_encoder,
        prediction_type='x0',
        time_rescaler=time_rescaler,
        input_rescaler=input_rescaler,
    )
    variables = layer.init(
        self.key,
        time=self.t,
        xt=self.xt,
        conditioning=self.conditioning,
        is_training=self.is_training,
    )
    params = variables['params']
    shifted_params = jax.tree.map(lambda x: x + 1e-4, params)
    # shift the params by a small amount to ensure that we are not sensitive to
    # zero initialization of the params impacting the conditioning embeddings.

    base_inference_fn = wrappers.FlaxLinenInferenceFn(
        network=layer, params=shifted_params
    )
    inference_fn = diffusion_inference.GuidedDiffusionInferenceFn(
        base_inference_fn=base_inference_fn,
        guidance_fn=guidance.ScalarGuidanceFn(guidance=0.0),
    )
    other_conditioning = {
        'label_foo': jnp.zeros_like(self.conditioning['label_foo']),
        'label_bar': jnp.zeros_like(self.conditioning['label_bar']),
    }

    output = inference_fn(
        xt=self.xt, conditioning=self.conditioning, time=self.t
    )
    other_output = inference_fn(
        xt=self.xt, conditioning=other_conditioning, time=self.t
    )

    self.assertIn('x0', output)
    self.assertIn('x0', other_output)
    self.assertFalse(jnp.allclose(output['x0'], other_output['x0']))

  def test_nested_inference(self):
    """Tests that the nested inference function works."""
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
    # shift the params by a small amount to ensure that we are not sensitive to
    # zero initialization of the params impacting the conditioning embeddings.
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
