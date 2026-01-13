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

"""Tests for discrete step sampler."""

import chex
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling import discrete_step_sampler
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = sampling_base.DiffusionStep
StepInfo = sampling_base.StepInfo
CategoricalProcess = discrete.CategoricalProcess
UnMaskingStep = discrete_step_sampler.UnMaskingStep

################################################################################
# MARK: Tests
################################################################################


class RemaskingFnTest(parameterized.TestCase):
  """Tests for remasking functions."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.s = jnp.array([0.1, 0.2])
    self.t = jnp.array([0.5, 0.6])

  def test_no_remasking(self):
    fn = discrete_step_sampler.NoRemaskingFn()
    result = fn(self.s, self.t)
    chex.assert_trees_all_close(result, jnp.zeros_like(self.s))

  @parameterized.named_parameters(
      ('invalid_switch', 0.5, 1.0, 0.0, ValueError),
      ('invalid_max_cap', -0.1, 0.0, 1.0, ValueError),
  )
  def test_max_capped_remasking_init_raises(
      self, max_cap, switch_min, switch_max, error
  ):
    with self.assertRaises(error):
      discrete_step_sampler.MaxCappedRemaskingFn(
          schedule=self.schedule,
          max_cap=max_cap,
          switch_min=switch_min,
          switch_max=switch_max,
      )

  def test_max_capped_remasking_call(self):
    fn = discrete_step_sampler.MaxCappedRemaskingFn(
        schedule=self.schedule, max_cap=0.8
    )
    alpha_s = self.schedule.alpha(self.s)
    alpha_t = self.schedule.alpha(self.t)
    expected = jnp.minimum((1.0 - alpha_s) / alpha_t, 0.8)
    result = fn(self.s, self.t)
    chex.assert_trees_all_close(result, expected)

  @parameterized.named_parameters(
      ('invalid_switch', 0.5, 1.0, 0.0, ValueError),
      ('invalid_rescale_factor_neg', -0.1, 0.0, 1.0, ValueError),
      ('invalid_rescale_factor_one', 1.0, 0.0, 1.0, ValueError),
  )
  def test_rescaled_remasking_init_raises(
      self, rescale_factor, switch_min, switch_max, error
  ):
    with self.assertRaises(error):
      discrete_step_sampler.RescaledRemaskingFn(
          schedule=self.schedule,
          rescale_factor=rescale_factor,
          switch_min=switch_min,
          switch_max=switch_max,
      )

  def test_rescaled_remasking_call(self):
    fn = discrete_step_sampler.RescaledRemaskingFn(
        schedule=self.schedule, rescale_factor=0.5
    )
    alpha_s = self.schedule.alpha(self.s)
    alpha_t = self.schedule.alpha(self.t)
    expected = 0.5 * jnp.minimum((1.0 - alpha_s) / alpha_t, 1.0)
    result = fn(self.s, self.t)
    chex.assert_trees_all_close(result, expected)


class CorruptedMaskFnTest(absltest.TestCase):
  """Tests for corrupted mask functions."""

  def setUp(self):
    super().setUp()
    self.xt = jnp.array([[[0], [1], [4]]])  # shape (1, 3, 1)
    self.schedule = schedules.LinearDiscreteSchedule()
    self.process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=4
    )  # mask_value is 4

  def test_all_corrupted_mask_fn(self):
    fn = discrete_step_sampler.AllCorruptedMaskFn()
    mask = fn(self.xt)
    expected_mask = jnp.ones_like(self.xt, dtype=jnp.bool_)
    chex.assert_trees_all_equal(mask, expected_mask)

  def test_mask_value_corrupted_mask_fn(self):
    fn = discrete_step_sampler.MaskValueCorruptedMaskFn(process=self.process)
    mask = fn(self.xt)
    expected_mask = jnp.array([[[False], [False], [True]]])
    chex.assert_trees_all_equal(mask, expected_mask)


class UnMaskingStepTest(absltest.TestCase):
  """Tests for the UnMaskingStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    self.initial_noise = jnp.ones((2, 4, 1), dtype=jnp.int32) * (
        self.process.process_num_categories - 1
    )
    self.unmasking_step = UnMaskingStep(corruption_process=self.process)

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 0.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32) - jnp.inf

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Test case 1: Full unmasking
    next_step_info_full_unmask = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_full_unmask = self.unmasking_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_full_unmask,
    )
    expected_xt_full_unmask = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_equal(
        next_step_full_unmask.xt, expected_xt_full_unmask
    )

    # Test case 2: No unmasking
    next_step_info_no_unmask = StepInfo(
        step=1,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_no_unmask = self.unmasking_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_no_unmask,
    )
    chex.assert_trees_all_equal(next_step_no_unmask.xt, initial_step.xt)

  def test_finalize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )
    last_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    final_step = self.unmasking_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )
    expected_xt = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_equal(final_step.xt, expected_xt)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError, 'UnMaskingStep only supports masking processes.'
    ):
      discrete_step_sampler.UnMaskingStep(corruption_process=process)


class DiscreteDDIMStepTest(absltest.TestCase):
  """Tests for the DiscreteDDIMStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.randint(
        key, (2, 4, 1), 0, self.process.process_num_categories
    )
    self.ddim_step = discrete_step_sampler.DiscreteDDIMStep(
        corruption_process=self.process
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 0.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.process_num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32) - jnp.inf

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step = self.ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    self.assertEqual(next_step.xt.shape, self.initial_noise.shape)
    self.assertEqual(next_step.xt.dtype, self.initial_noise.dtype)

  def test_finalize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    last_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    final_step = self.ddim_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )

    expected_xt = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_close(final_step.xt, expected_xt)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError, 'DiscreteDDIMStep does not support masking processes.'
    ):
      discrete_step_sampler.DiscreteDDIMStep(corruption_process=process)


if __name__ == '__main__':
  absltest.main()
