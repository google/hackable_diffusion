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

"""Tests for multimodal wrappers."""

from unittest import mock

import chex
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import multimodal
from hackable_diffusion.lib import time_sampling
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.inference import guidance
from hackable_diffusion.lib.inference import projection
from hackable_diffusion.lib.loss import discrete as discrete_loss
from hackable_diffusion.lib.loss import gaussian as gaussian_loss
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import discrete_step_sampler
from hackable_diffusion.lib.sampling import gaussian_step_sampler
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

Float = hd_typing.Float


def _create_leaf_process(data_array, time_array, target_info_name):
  process = mock.MagicMock()
  target_info = {target_info_name: data_array + 5}
  process.corrupt.return_value = (data_array + 5.0, target_info)
  process.sample_from_invariant.return_value = data_array - 1.0
  process.convert_predictions.return_value = target_info
  process.get_schedule_info.return_value = {'time': time_array - 7.0}
  return process


class IdentityBackbone(nn.Module, arch_typing.ConditionalBackbone):

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


################################################################################
# MARK: NestedProcess Tests
################################################################################


class NestedProcessTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('batch_size_1', 1),
      ('batch_size_16', 16),
  )
  def test_nested_process(self, batch_size: int = 16):
    data_tree = {
        'a': {
            'b': jnp.ones((batch_size, 1)),
            'c': {
                'd': jnp.ones((batch_size, 5, 7, 11)),
                'e': jnp.ones((batch_size, 13)),
            },
        },
    }
    time_tree = {
        'a': {
            'b': jnp.ones((batch_size, 1)) * 0.3,
            'c': {
                'd': jnp.ones((batch_size, 5, 7, 11)) * 0.5,
                'e': jnp.ones((batch_size, 13)) * 0.7,
            },
        },
    }
    target_info_tree_names = {
        'a': {
            'b': 'x0',
            'c': {'d': 'score', 'e': 'velocity'},
        },
    }
    processes_tree = jax.tree.map(
        _create_leaf_process, data_tree, time_tree, target_info_tree_names
    )
    key = jax.random.PRNGKey(0)
    nested_process = multimodal.NestedProcess(processes=processes_tree)
    xt, target_info = nested_process.corrupt(key, data_tree, time_tree)
    invariant_out = nested_process.sample_from_invariant(key, data_tree)
    convert_predictions_out = nested_process.convert_predictions(
        target_info, xt, time_tree
    )
    schedule_info = nested_process.get_schedule_info(time_tree)

    expected_invariant_out = {
        'a': {
            'b': jnp.ones((batch_size, 1)) - 1.0,
            'c': {
                'd': jnp.ones((batch_size, 5, 7, 11)) - 1.0,
                'e': jnp.ones((batch_size, 13)) - 1.0,
            },
        },
    }
    expected_convert_predictions_out = {
        'a': {
            'b': {'x0': jnp.ones((batch_size, 1)) + 5},
            'c': {
                'd': {'score': jnp.ones((batch_size, 5, 7, 11)) + 5},
                'e': {'velocity': jnp.ones((batch_size, 13)) + 5},
            },
        },
    }
    expected_schedule_info = {
        'a': {
            'b': {'time': jnp.ones((batch_size, 1)) * 0.3 - 7.0},
            'c': {
                'd': {'time': jnp.ones((batch_size, 5, 7, 11)) * 0.5 - 7.0},
                'e': {'time': jnp.ones((batch_size, 13)) * 0.7 - 7.0},
            },
        },
    }

    chex.assert_trees_all_close(expected_invariant_out, invariant_out)
    chex.assert_trees_all_close(
        expected_convert_predictions_out, convert_predictions_out
    )
    chex.assert_trees_all_close(expected_schedule_info, schedule_info)


################################################################################
# MARK: NestedSamplerStep Tests
################################################################################


class NestedSamplerStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.stochasticity_level = 1.0
    self.num_categories = 256

    schedule_continuous = schedules.RFSchedule()
    process_continuous = gaussian.GaussianProcess(schedule=schedule_continuous)

    schedule_discrete = schedules.CosineDiscreteSchedule()
    process_discrete = discrete.CategoricalProcess.masking_process(
        schedule=schedule_discrete, num_categories=self.num_categories
    )

    time_schedule = multimodal.NestedTimeSchedule(
        time_schedules={
            'data_continuous': time_scheduling.UniformTimeSchedule(),
            'modality': {
                'data_discrete': time_scheduling.UniformTimeSchedule()
            },
        }
    )
    self.time_schedule = time_schedule

    stepper_continuous = gaussian_step_sampler.DDIMStep(
        corruption_process=process_continuous,
        stoch_coeff=self.stochasticity_level,
    )
    stepper_discrete = discrete_step_sampler.UnMaskingStep(
        corruption_process=process_discrete
    )
    stepper = multimodal.NestedSamplerStep(
        sampler_steps={
            'data_continuous': stepper_continuous,
            'modality': {'data_discrete': stepper_discrete},
        }
    )
    self.stepper = stepper

    self.xt = {
        'data_continuous': jnp.zeros((1, 28, 28, 3)),
        'modality': {'data_discrete': jnp.zeros((1, 128, 1))},
    }
    self.prediction = {
        'data_continuous': {'x0': jnp.zeros((1, 28, 28, 3))},
        'modality': {
            'data_discrete': {
                'logits': jnp.zeros((1, 128, self.num_categories))
            }
        },
    }
    self.step_info = {
        'data_continuous': base.StepInfo(
            time=jnp.array((0.5,)),
            step=jnp.array(0),
            rng=jax.random.PRNGKey(0),
        ),
        'modality': {
            'data_discrete': base.StepInfo(
                time=jnp.array((0.5,)),
                step=jnp.array(0),
                rng=jax.random.PRNGKey(0),
            ),
        },
    }
    self.next_step_info = {
        'data_continuous': base.StepInfo(
            time=jnp.array((0.7,)),
            step=jnp.array(1),
            rng=jax.random.PRNGKey(1),
        ),
        'modality': {
            'data_discrete': base.StepInfo(
                time=jnp.array((0.7,)),
                step=jnp.array(1),
                rng=jax.random.PRNGKey(1),
            )
        },
    }
    self.diffusion_step = {
        'data_continuous': base.DiffusionStep(
            xt=self.xt['data_continuous'],
            step_info=self.step_info['data_continuous'],
            aux=dict(),
        ),
        'modality': {
            'data_discrete': base.DiffusionStep(
                xt=self.xt['modality']['data_discrete'],
                step_info=self.step_info['modality']['data_discrete'],
                aux={
                    'logits': self.prediction['modality']['data_discrete'][
                        'logits'
                    ]
                },
            )
        },
    }

  def test_nested_sampler_initialize(self):
    step = self.stepper.initialize(
        initial_noise=self.xt, initial_step_info=self.step_info
    )
    self.assertEqual(step['data_continuous'].xt.shape, (1, 28, 28, 3))
    self.assertEqual(step['modality']['data_discrete'].xt.shape, (1, 128, 1))
    chex.assert_trees_all_equal_structs(step, self.diffusion_step)

  def test_nested_sampler_update(self):
    step = self.stepper.update(
        prediction=self.prediction,
        current_step=self.diffusion_step,
        next_step_info=self.next_step_info,
    )
    self.assertEqual(step['data_continuous'].xt.shape, (1, 28, 28, 3))
    self.assertEqual(step['modality']['data_discrete'].xt.shape, (1, 128, 1))
    chex.assert_trees_all_equal_structs(step, self.diffusion_step)

  def test_nested_sampler_finalize(self):
    step = self.stepper.finalize(
        prediction=self.prediction,
        current_step=self.diffusion_step,
        last_step_info=self.step_info,
    )
    self.assertEqual(step['data_continuous'].xt.shape, (1, 28, 28, 3))
    self.assertEqual(step['modality']['data_discrete'].xt.shape, (1, 128, 1))
    chex.assert_trees_all_equal_structs(step, self.diffusion_step)


################################################################################
# MARK: NestedTimeSchedule Tests
################################################################################


class NestedTimeScheduleTest(absltest.TestCase):

  def test_nested_time_schedule(self):
    time_schedule = multimodal.NestedTimeSchedule(
        time_schedules={
            'data_continuous': time_scheduling.UniformTimeSchedule(),
            'modality': {
                'data_discrete': time_scheduling.UniformTimeSchedule()
            },
        }
    )

    data_spec = {
        'data_continuous': jnp.zeros((2, 3)),
        'modality': {'data_discrete': jnp.zeros((2, 4, 5))},
    }
    num_steps = 5
    time_info = time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=num_steps,
        data_spec=data_spec,
    )
    self.assertIsInstance(time_info, dict)
    self.assertEqual(time_info['data_continuous'].time.shape, (5, 2, 1))
    self.assertEqual(
        time_info['modality']['data_discrete'].time.shape, (5, 2, 1, 1)
    )


################################################################################
# MARK: NestedDiffusionLoss Tests
################################################################################


class NestedDiffusionLossTest(parameterized.TestCase):

  def test_nested_loss(self):
    loss_fn_continuous = gaussian_loss.NoWeightGaussianLoss()
    loss_fn_discrete = discrete_loss.NoWeightDiscreteLoss()
    loss_fn = multimodal.NestedDiffusionLoss(
        losses={
            'data_continuous': loss_fn_continuous,
            'data_discrete': loss_fn_discrete,
        }
    )

    time = {
        'data_continuous': jnp.ones((1,)) * 0.5,
        'data_discrete': jnp.ones((1,)) * 0.5,
    }
    targets = {
        'data_continuous': {'x0': jnp.ones((1, 32, 32, 3))},
        'data_discrete': {
            'x0': jnp.ones((1, 128, 1), dtype=jnp.int32),
            'logits': jnp.ones((1, 128, 64), dtype=jnp.int32),
        },
    }
    output = {
        'data_continuous': {'x0': jnp.ones((1, 32, 32, 3))},
        'data_discrete': {'logits': jnp.ones((1, 128, 64))},
    }

    loss_output = loss_fn(preds=output, targets=targets, time=time)

    chex.assert_trees_all_equal_structs(loss_output, time)


################################################################################
# MARK: NestedGuidanceFn Tests
################################################################################


class NestedGuidanceFnTest(parameterized.TestCase):

  def test_nested_guidance_fn(self):
    batch_size = 2
    data_shape = (4, 4, 3)
    xt_data = jnp.ones((batch_size, *data_shape))
    guidance_val = 3.0
    guidance_fn = multimodal.NestedGuidanceFn(
        guidance_fns={
            'data_continuous': guidance.ScalarGuidanceFn(guidance=guidance_val),
            'modality': {
                'data_discrete': guidance.ScalarGuidanceFn(
                    guidance=guidance_val
                ),
            },
        }
    )
    nested_xt = {
        'data_continuous': xt_data,
        'modality': {'data_discrete': xt_data},
    }
    time = {
        'data_continuous': jnp.array([0.5, 0.5]),
        'modality': {'data_discrete': jnp.array([0.5, 0.5])},
    }
    cond_outputs = {
        'data_continuous': {'pred': jnp.ones_like(xt_data) * 2.0},
        'modality': {'data_discrete': {'pred': jnp.ones_like(xt_data) * 2.0}},
    }
    uncond_outputs = {
        'data_continuous': {'pred': jnp.ones_like(xt_data) * 1.0},
        'modality': {'data_discrete': {'pred': jnp.ones_like(xt_data) * 1.0}},
    }

    result = guidance_fn(nested_xt, {}, time, cond_outputs, uncond_outputs)
    self.assertIsInstance(result, dict)
    self.assertEqual(
        result['data_continuous']['pred'].shape,
        (batch_size, *data_shape),
    )
    self.assertEqual(
        result['modality']['data_discrete']['pred'].shape,
        (batch_size, *data_shape),
    )


################################################################################
# MARK: NestedProjectionFn Tests
################################################################################


class NestedProjectionFnTest(parameterized.TestCase):

  def test_nested_projection_fn(self):
    batch_size = 2
    data_shape = (4, 4, 3)
    other_data_shape = (4, 8, 9)
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())

    nested_xt = {
        'data_continuous_1': jnp.ones((batch_size, *data_shape)),
        'modality': {
            'data_continuous_2': jnp.ones((batch_size, *other_data_shape))
        },
    }
    nested_time = {
        'data_continuous_1': utils.bcast_right(
            jnp.array([0.5, 0.5]), nested_xt['data_continuous_1'].ndim
        ),
        'modality': {
            'data_continuous_2': utils.bcast_right(
                jnp.array([0.5, 0.5]),
                nested_xt['modality']['data_continuous_2'].ndim,
            )
        },
    }
    nested_outputs = {
        'data_continuous_1': {
            'x0': jnp.ones_like(nested_xt['data_continuous_1'])
        },
        'modality': {
            'data_continuous_2': {
                'x0': jnp.ones_like(nested_xt['modality']['data_continuous_2'])
            }
        },
    }

    proj_fn = multimodal.NestedProjectionFn(
        projection_fns={
            'data_continuous_1': projection.IdentityProjectionFn(),
            'modality': {
                'data_continuous_2': projection.StaticThresholdProjectionFn(
                    process=process
                )
            },
        }
    )
    result = proj_fn(nested_xt, {}, nested_time, nested_outputs)
    self.assertIsInstance(result, dict)
    self.assertEqual(
        result['data_continuous_1']['x0'].shape,
        nested_xt['data_continuous_1'].shape,
    )
    self.assertEqual(
        result['modality']['data_continuous_2']['x0'].shape,
        nested_xt['modality']['data_continuous_2'].shape,
    )
    chex.assert_trees_all_equal_structs(result, nested_outputs)


################################################################################
# MARK: NestedTimeSampler Tests
################################################################################


class NestedTimeSamplerTest(parameterized.TestCase):

  def test_nested_time_sampler(self):
    key = jax.random.PRNGKey(0)
    data_spec = {
        'image': jnp.zeros((2, 3, 4)),
        'modality': {'label': jnp.zeros((2,))},
    }

    sampler = multimodal.NestedTimeSampler(
        samplers={
            'image': time_sampling.UniformTimeSampler(axes=(0, 1)),
            'modality': {'label': time_sampling.UniformTimeSampler()},
        }
    )
    time = sampler(key, data_spec)

    self.assertIsInstance(time, dict)
    self.assertEqual(time['image'].shape, (2, 3, 1))
    self.assertEqual(time['modality']['label'].shape, (2,))

  def test_joint_nested_time_sampler(self):
    key = jax.random.PRNGKey(0)
    data_spec = {
        'image': jnp.zeros((2, 3, 4)),
        'modality': {'label': jnp.zeros((2,))},
    }

    sampler = multimodal.JointNestedTimeSampler(
        samplers={
            'image': time_sampling.UniformTimeSampler(),
            'modality': {'label': time_sampling.UniformTimeSampler()},
        }
    )
    time = sampler(key, data_spec)

    self.assertIsInstance(time, dict)
    self.assertEqual(time['image'].shape, (2, 1, 1))
    self.assertEqual(time['modality']['label'].shape, (2,))
    chex.assert_trees_all_close(
        time['image'].squeeze(),
        time['modality']['label'].squeeze(),
    )
    chex.assert_trees_all_equal_structs(time, data_spec)


if __name__ == '__main__':
  absltest.main()
