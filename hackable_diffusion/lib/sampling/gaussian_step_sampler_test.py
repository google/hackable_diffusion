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

"""Tests for Gaussian step sampler."""

import itertools

import chex
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling import gaussian_step_sampler
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = sampling_base.DiffusionStep
StepInfo = sampling_base.StepInfo
GaussianProcess = gaussian.GaussianProcess
TargetInfo = hd_typing.TargetInfo

################################################################################
# MARK: Helper functions
################################################################################

dummy_inference_fn = lambda xt, conditioning, time: {'x0': xt}


def _sde_update(
    xt: jnp.ndarray,
    prediction: TargetInfo,
    time: jnp.ndarray,
    next_time: jnp.ndarray,
    stochasticity_level: float,
    process: GaussianProcess,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Helper function to compute the SDE update."""
  f = process.schedule.f(time)
  g = process.schedule.g(time)
  score = process.convert_predictions(
      prediction=prediction,
      xt=xt,
      time=time,
  )['score']
  dt = time - next_time
  delta = (
      -f * xt
      + 0.5 * jnp.square(g) * (1.0 + jnp.square(stochasticity_level)) * score
  )
  mean = xt + delta * dt
  volatility = jnp.sqrt(dt) * g * stochasticity_level
  return mean, volatility


def _ddim_update(
    xt: jnp.ndarray,
    prediction: TargetInfo,
    time: jnp.ndarray,
    next_time: jnp.ndarray,
    stochasticity_level: float,
    process: GaussianProcess,
):
  """Helper function to compute the DDIM update."""
  x0 = process.convert_predictions(
      prediction=prediction,
      xt=xt,
      time=time,
  )['x0']

  alpha = process.schedule.alpha(time)
  sigma = process.schedule.sigma(time)
  next_alpha = process.schedule.alpha(next_time)
  next_sigma = process.schedule.sigma(next_time)

  # alpha sigma ratios
  r01 = next_sigma / sigma
  r11 = alpha / next_alpha * r01
  r12 = r11 * r01
  r22 = r11 * r11

  # DDIM update
  coeff_xt = stochasticity_level * r12 + (1.0 - stochasticity_level) * r01
  coeff_x0 = next_alpha * (
      1.0 - stochasticity_level * r22 - (1.0 - stochasticity_level) * r11
  )
  volatility = next_sigma * jnp.sqrt(
      1.0 - jnp.square(stochasticity_level * r11 + (1.0 - stochasticity_level))
  )
  new_mean = coeff_xt * xt + coeff_x0 * x0
  return new_mean, volatility


################################################################################
# MARK: Test constants
################################################################################

_STOCHASTICITY_LEVELS = (0.0, 0.5, 1.0)
_USE_STOCHASTIC_LAST_STEP = (False, True)


################################################################################
# MARK: Tests
################################################################################


class SdeStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.schedule = schedules.RFSchedule()
    self.process = gaussian.GaussianProcess(schedule=self.schedule)
    self.initial_noise = jnp.expand_dims(jnp.eye(4), axis=0)

  @parameterized.parameters(
      itertools.product(_STOCHASTICITY_LEVELS, _USE_STOCHASTIC_LAST_STEP)
  )
  def test_initialize(self, stochasticity_level, use_stochastic_last_step):

    sde_step = gaussian_step_sampler.SdeStep(
        corruption_process=self.process,
        churn=stochasticity_level,
        stochastic_last_step=use_stochastic_last_step,
    )

    initial_step = sde_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0, time=jnp.array([0.0]), rng=jax.random.PRNGKey(0)
        ),
    )

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=jnp.expand_dims(jnp.eye(4), axis=0),
            step_info=StepInfo(
                step=0,
                time=jnp.array([0.0]),
                rng=jax.random.PRNGKey(0),
            ),
            aux=dict(),
        ),
    )

  @parameterized.parameters(
      itertools.product(_STOCHASTICITY_LEVELS, _USE_STOCHASTIC_LAST_STEP)
  )
  def test_update(self, stochasticity_level, use_stochastic_last_step):

    sde_step = gaussian_step_sampler.SdeStep(
        corruption_process=self.process,
        churn=stochasticity_level,
        stochastic_last_step=use_stochastic_last_step,
    )

    initial_step = sde_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1]),
        rng=jax.random.PRNGKey(1),
    )

    next_step = sde_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )
    z = jax.random.normal(key=next_step_info.rng, shape=initial_step.xt.shape)
    mean, volatility = _sde_update(
        xt=initial_step.xt,
        prediction=prediction,
        time=initial_step.step_info.time,
        next_time=next_step_info.time,
        stochasticity_level=stochasticity_level,
        process=self.process,
    )
    expected_xt = mean + volatility * z

    chex.assert_trees_all_close(
        next_step,
        DiffusionStep(
            xt=expected_xt,
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )

  @parameterized.parameters(
      itertools.product(_STOCHASTICITY_LEVELS, _USE_STOCHASTIC_LAST_STEP)
  )
  def test_finalize(self, stochasticity_level, use_stochastic_last_step):
    sde_step = gaussian_step_sampler.SdeStep(
        corruption_process=self.process,
        churn=stochasticity_level,
        stochastic_last_step=use_stochastic_last_step,
    )

    initial_step = sde_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    last_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1]),
        rng=jax.random.PRNGKey(1),
    )

    final_step = sde_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )

    z = jax.random.normal(key=last_step_info.rng, shape=initial_step.xt.shape)
    mean, volatility = _sde_update(
        xt=initial_step.xt,
        prediction=prediction,
        time=initial_step.step_info.time,
        next_time=last_step_info.time,
        stochasticity_level=stochasticity_level,
        process=self.process,
    )
    if use_stochastic_last_step:
      expected_xt = mean + volatility * z
    else:
      expected_xt = mean

    chex.assert_trees_all_close(
        final_step,
        DiffusionStep(
            xt=expected_xt,
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )

  def test_update_specific_parameters(self):

    sde_step = gaussian_step_sampler.SdeStep(
        corruption_process=self.process,
        churn=0.1,
    )

    initial_step = sde_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step = sde_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1]),
            rng=jax.random.PRNGKey(1),
        ),
    )

    chex.assert_trees_all_close(
        next_step,
        DiffusionStep(
            xt=jnp.array(
                [[
                    [0.995297, 0.001894, -0.003041, -0.003467],
                    [0.028324, 1.002066, 0.047887, 0.02242],
                    [-0.006492, 0.008013, 0.98292, -0.005491],
                    [0.019802, 0.017578, 0.019877, 1.011033],
                ]],
                dtype=jnp.float32,
            ),
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )


class AdjustedDDIMStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.schedule = schedules.RFSchedule()
    self.process = gaussian.GaussianProcess(schedule=self.schedule)
    self.initial_noise = jnp.expand_dims(jnp.eye(4), axis=0)

    self.adjusted_ddim_step = gaussian_step_sampler.AdjustedDDIMStep(
        corruption_process=self.process,
    )

  def test_initialize(self):

    initial_step = self.adjusted_ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0, time=jnp.array([0.0]), rng=jax.random.PRNGKey(0)
        ),
    )

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=jnp.expand_dims(jnp.eye(4), axis=0),
            step_info=StepInfo(
                step=0,
                time=jnp.array([0.0]),
                rng=jax.random.PRNGKey(0),
            ),
            aux=dict(),
        ),
    )

  def test_update(self):

    initial_step = self.adjusted_ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step = self.adjusted_ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1]),
            rng=jax.random.PRNGKey(1),
        ),
    )

    chex.assert_trees_all_close(
        next_step,
        DiffusionStep(
            xt=jnp.array(
                [[
                    [0.924722, 0.0, 0.0, 0.0],
                    [0.0, 0.924722, 0.0, 0.0],
                    [0.0, 0.0, 0.924722, 0.0],
                    [0.0, 0.0, 0.0, 0.924722],
                ]],
                dtype=jnp.float32,
            ),
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )

  def test_finalize(self):
    initial_step = self.adjusted_ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    final_step = self.adjusted_ddim_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1]),
            rng=jax.random.PRNGKey(1),
        ),
    )

    chex.assert_trees_all_close(
        final_step,
        DiffusionStep(
            xt=jnp.array(
                [[
                    [0.924722, 0.0, 0.0, 0.0],
                    [0.0, 0.924722, 0.0, 0.0],
                    [0.0, 0.0, 0.924722, 0.0],
                    [0.0, 0.0, 0.0, 0.924722],
                ]],
                dtype=jnp.float32,
            ),
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )


class DDIMStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.schedule = schedules.RFSchedule()
    self.process = gaussian.GaussianProcess(schedule=self.schedule)
    self.initial_noise = jnp.expand_dims(jnp.eye(4), axis=0)

  @parameterized.parameters(
      itertools.product(_STOCHASTICITY_LEVELS, _USE_STOCHASTIC_LAST_STEP)
  )
  def test_initialize(self, stochasticity_level, use_stochastic_last_step):

    ddim_step = gaussian_step_sampler.DDIMStep(
        corruption_process=self.process,
        stoch_coeff=stochasticity_level,
        stochastic_last_step=use_stochastic_last_step,
    )

    initial_step = ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.0]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=jnp.expand_dims(jnp.eye(4), axis=0),
            step_info=StepInfo(
                step=0,
                time=jnp.array([0.0]),
                rng=jax.random.PRNGKey(0),
            ),
            aux=dict(),
        ),
    )

  @parameterized.parameters(
      itertools.product(_STOCHASTICITY_LEVELS, _USE_STOCHASTIC_LAST_STEP)
  )
  def test_update(self, stochasticity_level, use_stochastic_last_step):

    ddim_step = gaussian_step_sampler.DDIMStep(
        corruption_process=self.process,
        stoch_coeff=stochasticity_level,
        stochastic_last_step=use_stochastic_last_step,
    )

    initial_step = ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step = ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1, time=jnp.array([0.1]), rng=jax.random.PRNGKey(1)
        ),
    )

    z = jax.random.normal(
        key=next_step.step_info.rng, shape=initial_step.xt.shape
    )

    mean, volatility = _ddim_update(
        xt=initial_step.xt,
        prediction=prediction,
        time=initial_step.step_info.time,
        next_time=next_step.step_info.time,
        stochasticity_level=stochasticity_level,
        process=self.process,
    )
    expected_xt = mean + volatility * z

    chex.assert_trees_all_close(
        next_step,
        DiffusionStep(
            xt=expected_xt,
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )

  @parameterized.parameters(
      itertools.product(_STOCHASTICITY_LEVELS, _USE_STOCHASTIC_LAST_STEP)
  )
  def test_finalize(self, stochasticity_level, use_stochastic_last_step):
    ddim_step = gaussian_step_sampler.DDIMStep(
        corruption_process=self.process,
        stoch_coeff=stochasticity_level,
        stochastic_last_step=use_stochastic_last_step,
    )

    initial_step = ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    final_step = ddim_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=StepInfo(
            step=1, time=jnp.array([0.1]), rng=jax.random.PRNGKey(1)
        ),
    )

    z = jax.random.normal(
        key=final_step.step_info.rng, shape=initial_step.xt.shape
    )

    mean, volatility = _ddim_update(
        xt=initial_step.xt,
        prediction=prediction,
        time=initial_step.step_info.time,
        next_time=final_step.step_info.time,
        stochasticity_level=stochasticity_level,
        process=self.process,
    )

    if use_stochastic_last_step:
      expected_xt = mean + volatility * z
    else:
      expected_xt = mean

    chex.assert_trees_all_close(
        final_step,
        DiffusionStep(
            xt=expected_xt,
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )

  def test_update_specific_parameters(self):

    ddim_step = gaussian_step_sampler.DDIMStep(
        corruption_process=self.process,
        stoch_coeff=0.25,
    )

    initial_step = ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(0),
        ),
    )

    prediction = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step = ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1, time=jnp.array([0.1]), rng=jax.random.PRNGKey(1)
        ),
    )

    chex.assert_trees_all_close(
        next_step,
        DiffusionStep(
            xt=jnp.array(
                [[
                    [0.978259, 0.004307, -0.006913, -0.007882],
                    [0.064399, 0.993651, 0.108881, 0.050978],
                    [-0.014761, 0.018219, 0.950119, -0.012484],
                    [0.045024, 0.039968, 0.045195, 1.01404],
                ]],
                dtype=jnp.float32,
            ),
            step_info=StepInfo(
                step=1,
                time=jnp.array([0.1]),
                rng=jax.random.PRNGKey(1),
            ),
            aux={},
        ),
        atol=1e-6,
    )


class HeunStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.schedule = schedules.RFSchedule()
    self.process = gaussian.GaussianProcess(schedule=self.schedule)
    self.initial_noise = jnp.expand_dims(jnp.eye(4), axis=0)
    self.time_schedule = time_scheduling.UniformTimeSchedule()
    self.num_steps = 10

    self.heun_step = gaussian_step_sampler.HeunStep(
        corruption_process=self.process,
        time_schedule=self.time_schedule,
        num_steps=self.num_steps,
    )

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0, time=jnp.array([1.0]), rng=jax.random.PRNGKey(0)
    )
    initial_step = self.heun_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    # Check the main diffusion step.
    chex.assert_trees_all_equal(initial_step.xt, self.initial_noise)
    chex.assert_trees_all_equal(initial_step.step_info, initial_step_info)

    # Check the aux dictionary.
    self.assertEqual(initial_step.aux['internal_counter'], 0)
    chex.assert_trees_all_equal(
        initial_step.aux['current_update'].xt,
        self.initial_noise,
    )
    chex.assert_trees_all_equal(
        initial_step.aux['current_update'].step_info,
        initial_step_info,
    )

  def test_update(self):
    # We need to manually get the step infos for the test.
    all_step_infos = self.time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=self.num_steps,
        data_spec=self.initial_noise,
    )

    def get_step_info(step):
      return jax.tree.map(lambda x: x[step], all_step_infos)

    # Initialize at step 8 (time=0.2).
    initial_step_info = get_step_info(8)
    initial_step = self.heun_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    # First update (internal step 0 -> 1).
    prediction1 = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )
    next_step_info1 = get_step_info(9)
    intermediate_step = self.heun_step.update(
        prediction=prediction1,
        current_step=initial_step,
        next_step_info=next_step_info1,
    )

    # The returned step_info should be for step 10.
    chex.assert_trees_all_equal(
        intermediate_step.step_info.time, get_step_info(10).time
    )
    self.assertEqual(intermediate_step.aux['internal_counter'], 1)

    # Second update (internal step 1 -> 0).
    prediction2 = dummy_inference_fn(
        xt=intermediate_step.xt,
        conditioning={},
        time=intermediate_step.step_info.time,
    )
    next_step_info2 = get_step_info(10)
    final_step = self.heun_step.update(
        prediction=prediction2,
        current_step=intermediate_step,
        next_step_info=next_step_info2,
    )

    # Check final state.
    self.assertEqual(final_step.aux['internal_counter'], 0)
    chex.assert_trees_all_close(
        final_step.xt,
        jnp.array(
            [[
                [0.999262, 0.0, 0.0, 0.0],
                [0.0, 0.999262, 0.0, 0.0],
                [0.0, 0.0, 0.999262, 0.0],
                [0.0, 0.0, 0.0, 0.999262],
            ]],
            dtype=jnp.float32,
        ),
        atol=1e-5,
    )

  def test_fail_for_odd_num_steps(self):
    with self.assertRaisesRegex(ValueError, 'should be divisible by 2'):
      gaussian_step_sampler.HeunStep(
          corruption_process=self.process,
          time_schedule=self.time_schedule,
          num_steps=3,
      )

  def test_finalize(self):
    # We need to manually get the step infos for the test.
    all_step_infos = self.time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=self.num_steps,
        data_spec=self.initial_noise,
    )

    def get_step_info(step):
      return jax.tree.map(lambda x: x[step], all_step_infos)

    # Initialize at step 8 (time=0.2).
    initial_step_info = get_step_info(8)
    initial_step = self.heun_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    # First update (internal step 0 -> 1).
    prediction1 = dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )
    next_step_info1 = get_step_info(9)
    intermediate_step = self.heun_step.update(
        prediction=prediction1,
        current_step=initial_step,
        next_step_info=next_step_info1,
    )

    # The returned step_info should be for step 10.
    chex.assert_trees_all_equal(
        intermediate_step.step_info.time, get_step_info(10).time
    )
    self.assertEqual(intermediate_step.aux['internal_counter'], 1)

    # Second update (internal step 1 -> 0).
    prediction2 = dummy_inference_fn(
        xt=intermediate_step.xt,
        conditioning={},
        time=intermediate_step.step_info.time,
    )
    next_step_info2 = get_step_info(10)
    final_step = self.heun_step.finalize(
        prediction=prediction2,
        current_step=intermediate_step,
        last_step_info=next_step_info2,
    )

    # Check final state.
    self.assertEqual(final_step.aux['internal_counter'], 0)
    chex.assert_trees_all_close(
        final_step.xt,
        jnp.array(
            [[
                [0.999262, 0.0, 0.0, 0.0],
                [0.0, 0.999262, 0.0, 0.0],
                [0.0, 0.0, 0.999262, 0.0],
                [0.0, 0.0, 0.0, 0.999262],
            ]],
            dtype=jnp.float32,
        ),
        atol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
