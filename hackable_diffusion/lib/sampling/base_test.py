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

"""Test for base sampling functionalities."""

import chex
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import discrete_step_sampler
from hackable_diffusion.lib.sampling import gaussian_step_sampler
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


class BaseSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.stochasticity_level = 1.0  # Stochasticity coefficient in DDIM
    self.num_categories = 256

    # Create a continuous process
    schedule_continuous = schedules.RFSchedule()
    process_continuous = gaussian.GaussianProcess(schedule=schedule_continuous)

    schedule_discrete = schedules.CosineDiscreteSchedule()
    process_discrete = discrete.CategoricalProcess.masking_process(
        schedule=schedule_discrete, num_categories=self.num_categories
    )

    # Create a nested time schedule
    time_schedule_continuous = time_scheduling.UniformTimeSchedule()
    time_schedule_discrete = time_scheduling.UniformTimeSchedule()
    time_schedule = time_scheduling.NestedTimeSchedule(
        time_schedules={
            'data_continuous': time_schedule_continuous,
            'modality': {'data_discrete': time_schedule_discrete},
        }
    )
    self.time_schedule = time_schedule

    # Create a nested stepper
    stepper_continuous = gaussian_step_sampler.DDIMStep(
        corruption_process=process_continuous,
        stoch_coeff=self.stochasticity_level,
    )
    stepper_discrete = discrete_step_sampler.UnMaskingStep(
        corruption_process=process_discrete
    )
    stepper = base.NestedSamplerStep(
        sampler_steps={
            'data_continuous': stepper_continuous,
            'modality': {'data_discrete': stepper_discrete},
        }
    )
    self.stepper = stepper

    # variables
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
                aux=dict(),
            )
        },
    }

  def test_nested_sampler_initialize(self):
    """Test the nested sampler step."""
    step = self.stepper.initialize(
        initial_noise=self.xt, initial_step_info=self.step_info
    )
    self.assertEqual(step['data_continuous'].xt.shape, (1, 28, 28, 3))
    self.assertEqual(step['modality']['data_discrete'].xt.shape, (1, 128, 1))
    chex.assert_trees_all_equal_structs(step, self.diffusion_step)

  def test_nested_sampler_update(self):
    """Test the nested sampler step."""
    step = self.stepper.update(
        prediction=self.prediction,
        current_step=self.diffusion_step,
        next_step_info=self.next_step_info,
    )
    self.assertEqual(step['data_continuous'].xt.shape, (1, 28, 28, 3))
    self.assertEqual(step['modality']['data_discrete'].xt.shape, (1, 128, 1))
    chex.assert_trees_all_equal_structs(step, self.diffusion_step)

  def test_nested_sampler_finalize(self):
    """Test the nested sampler step."""
    step = self.stepper.finalize(
        prediction=self.prediction,
        current_step=self.diffusion_step,
        last_step_info=self.step_info,
    )
    self.assertEqual(step['data_continuous'].xt.shape, (1, 28, 28, 3))
    self.assertEqual(step['modality']['data_discrete'].xt.shape, (1, 128, 1))
    chex.assert_trees_all_equal_structs(step, self.diffusion_step)


if __name__ == '__main__':
  absltest.main()
