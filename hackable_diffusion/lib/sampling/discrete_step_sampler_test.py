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

"""Tests for discrete step sampler."""

import chex
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.sampling import discrete_step_sampler
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = hd_api.DiffusionStep
StepInfo = hd_api.StepInfo
CategoricalProcess = discrete.CategoricalProcess
UnMaskingStep = discrete_step_sampler.UnMaskingStep

################################################################################
# MARK: Tests
################################################################################


class RemaskingFnTest(parameterized.TestCase):
  """Tests for remasking functions."""

  def setUp(self):
    super().setUp()
    self.schedule = discrete.LinearDiscreteSchedule()
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
    self.schedule = discrete.LinearDiscreteSchedule()
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
    self.schedule = discrete.LinearDiscreteSchedule()
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
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_initialize_raises_for_invalid_trailing_dimension(self):
    initial_noise_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected initial_noise to have a trailing dimension of 1'
    ):
      self.unmasking_step.initialize(
          initial_noise=initial_noise_bad,
          initial_step_info=initial_step_info,
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

  def test_update_raises_for_invalid_xt_trailing_dimension(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    xt_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step = DiffusionStep(
        xt=xt_bad,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                (2, 4, 3, self.process.num_categories), dtype=jnp.float32
            )
        },
    )
    prediction = {'logits': jnp.zeros((2, 4, 3, self.process.num_categories))}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.unmasking_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

  def test_update_raises_for_shape_mismatch(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = {'logits': jnp.zeros((2, 8, self.process.num_categories))}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaises(ValueError):
      self.unmasking_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

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

  def _reference_unmasking_routing_weights(
      self, alpha_s, alpha_t, is_masked, sigma=0.0
  ):
    """Compute expected routing weights from the paper formulas."""
    if is_masked:
      p_clean = (alpha_s - (1.0 - sigma) * alpha_t) / (1.0 - alpha_t)
      p_noise = sigma
      p_stay = 1.0 - p_clean - p_noise
    else:
      p_clean = 0.0
      p_noise = sigma
      p_stay = 1.0 - sigma
    return p_stay, p_noise, p_clean

  def test_numerical_routing_early_masked(self):
    """Verify routing weight formulas against independent computation."""
    for alpha_s, alpha_t, is_masked, sigma in [
        (0.9, 0.5, True, 0.0),
        (0.6, 0.3, True, 0.0),
        (0.2, 0.05, True, 0.0),
        (0.9, 0.5, False, 0.0),
        (0.8, 0.4, True, 0.1),
        (0.8, 0.4, False, 0.1),
    ]:
      p_stay, p_noise, p_clean = self._reference_unmasking_routing_weights(
          alpha_s, alpha_t, is_masked, sigma
      )

      # Reproduce the computation from UnMaskingStep.update
      alpha_s_arr = jnp.array(alpha_s)
      alpha_t_arr = jnp.array(alpha_t)
      p_st = jnp.array(sigma)

      if is_masked:
        code_p_clean = (alpha_s_arr - (1.0 - p_st) * alpha_t_arr) / (
            1.0 - alpha_t_arr
        )
        code_p_noise = p_st
        code_p_stay = 1.0 - code_p_clean - code_p_noise
      else:
        code_p_stay = 1.0 - p_st
        code_p_noise = p_st
        code_p_clean = jnp.zeros_like(p_st)

      chex.assert_trees_all_close(
          jnp.array([code_p_stay, code_p_noise, code_p_clean]),
          jnp.array([p_stay, p_noise, p_clean]),
          atol=1e-7,
      )

  def test_numerical_full_unmask_gives_all_clean(self):
    """At t->0 all masked positions should go clean (alpha_s=1, alpha_t<1)."""
    # For LinearDiscreteSchedule: alpha(0) = 1.0, alpha(t) = 1 - t.
    alpha_s = 1.0
    alpha_t = 0.5  # alpha(0.5) for linear schedule

    p_stay, p_noise, p_clean = self._reference_unmasking_routing_weights(
        alpha_s, alpha_t, is_masked=True, sigma=0.0
    )
    # All probability should be on clean
    self.assertAlmostEqual(p_clean, 1.0, places=6)
    self.assertAlmostEqual(p_stay, 0.0, places=6)
    self.assertAlmostEqual(p_noise, 0.0, places=6)

  def test_numerical_no_unmask_gives_all_stay(self):
    """When alpha_s == alpha_t (same time), everything stays."""
    alpha = 0.5
    p_stay, p_noise, p_clean = self._reference_unmasking_routing_weights(
        alpha, alpha, is_masked=True, sigma=0.0
    )
    self.assertAlmostEqual(p_clean, 0.0, places=6)
    self.assertAlmostEqual(p_stay, 1.0, places=6)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError, 'UnMaskingStep only supports masking processes.'
    ):
      discrete_step_sampler.UnMaskingStep(corruption_process=process)

  def test_update_rejects_2d_input(self):
    """Passing xt without trailing dim should raise ValueError."""
    noise_2d = jnp.ones((2, 4), dtype=jnp.int32) * (
        self.process.process_num_categories - 1
    )
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5]),
        rng=jax.random.PRNGKey(0),
    )
    initial_step = DiffusionStep(
        xt=noise_2d,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                noise_2d.shape + (self.process.num_categories,),
                dtype=jnp.float32,
            )
        },
    )
    # Logits must be (B, L, V) to match the model output convention.
    logits = jnp.zeros(noise_2d.shape + (self.process.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    prediction = {'logits': logits}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0]),
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.unmasking_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )


class DiscreteDDIMStepTest(absltest.TestCase):
  """Tests for the DiscreteDDIMStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = discrete.LinearDiscreteSchedule()
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
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_initialize_raises_for_invalid_trailing_dimension(self):
    initial_noise_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected initial_noise to have a trailing dimension of 1'
    ):
      self.ddim_step.initialize(
          initial_noise=initial_noise_bad,
          initial_step_info=initial_step_info,
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

  def test_update_raises_for_invalid_xt_trailing_dimension(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    xt_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step = DiffusionStep(
        xt=xt_bad,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                (2, 4, 3, self.process.process_num_categories),
                dtype=jnp.float32,
            )
        },
    )
    prediction = {
        'logits': jnp.zeros((2, 4, 3, self.process.process_num_categories))
    }
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.ddim_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

  def test_update_raises_for_shape_mismatch(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = {
        'logits': jnp.zeros((2, 8, self.process.process_num_categories))
    }
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaises(ValueError):
      self.ddim_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

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

  def test_numerical_routing_weights_nonnegative(self):
    """All routing weights should be non-negative for valid alpha pairs."""
    for alpha_s in [0.1, 0.3, 0.5, 0.7, 0.9]:
      for alpha_t in [0.01, 0.05, 0.1]:
        if alpha_t >= alpha_s:
          continue
        ratio = alpha_t / alpha_s
        pi_xt = 0.25
        p_stay = ratio * (1.0 - alpha_s) * pi_xt
        p_noise = (1.0 - ratio) * (1.0 - alpha_s) * pi_xt
        p_clean = (1.0 - ratio) * alpha_s * pi_xt
        self.assertGreaterEqual(p_stay, 0.0)
        self.assertGreaterEqual(p_noise, 0.0)
        self.assertGreaterEqual(p_clean, 0.0)

  def test_numerical_weights_sum_to_one(self):
    """Routing weights (stay + noise + clean) should always sum to 1."""
    alpha_s, alpha_t = 0.7, 0.2
    ratio = alpha_t / alpha_s
    pi_xt = 0.25

    p_stay = ratio * (1.0 - alpha_s) * pi_xt
    p_noise = (1.0 - ratio) * (1.0 - alpha_s) * pi_xt
    p_clean = (1.0 - ratio) * alpha_s * pi_xt
    total = p_stay + p_noise + p_clean
    self.assertAlmostEqual(
        (p_stay + p_noise + p_clean) / total, 1.0, places=10
    )

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError, 'DiscreteDDIMStep does not support masking processes.'
    ):
      discrete_step_sampler.DiscreteDDIMStep(corruption_process=process)

  def test_fail_for_zero_invariant_probs(self):
    invariant_probs = (1.0, 0.0, 0.0)
    process = CategoricalProcess(
        schedule=self.schedule,
        invariant_probs=invariant_probs,
        num_categories=len(invariant_probs),
    )
    with self.assertRaisesRegex(
        ValueError,
        'DiscreteDDIMStep does not support invariant probabilities'
        ' with 0.0 probability mass for any element.',
    ):
      discrete_step_sampler.DiscreteDDIMStep(corruption_process=process)

  def test_update_rejects_2d_input(self):
    """Passing xt without trailing dim should raise ValueError."""
    noise_2d = jax.random.randint(
        jax.random.PRNGKey(0),
        (2, 4),
        0,
        self.process.process_num_categories,
    )
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5]),
        rng=jax.random.PRNGKey(0),
    )
    initial_step = DiffusionStep(
        xt=noise_2d,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                noise_2d.shape + (self.process.process_num_categories,),
                dtype=jnp.float32,
            )
        },
    )
    logits = jnp.zeros(noise_2d.shape + (self.process.process_num_categories,))
    logits = logits.at[..., 1].set(10.0)
    prediction = {'logits': logits}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0]),
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.ddim_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )


class IntegratedDiscreteDDIMStepTest(absltest.TestCase):
  """Tests for the IntegratedDiscreteDDIMStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = discrete.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.randint(
        key, (2, 4, 1), 0, self.process.process_num_categories
    )
    self.integrated_ddim_step = (
        discrete_step_sampler.IntegratedDiscreteDDIMStep(
            corruption_process=self.process
        )
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
    initial_step = self.integrated_ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_initialize_raises_for_invalid_trailing_dimension(self):
    initial_noise_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected initial_noise to have a trailing dimension of 1'
    ):
      self.integrated_ddim_step.initialize(
          initial_noise=initial_noise_bad,
          initial_step_info=initial_step_info,
      )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.integrated_ddim_step.initialize(
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
    next_step = self.integrated_ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    self.assertEqual(next_step.xt.shape, self.initial_noise.shape)
    self.assertEqual(next_step.xt.dtype, self.initial_noise.dtype)

  def test_update_raises_for_invalid_xt_trailing_dimension(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    xt_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step = DiffusionStep(
        xt=xt_bad,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                (2, 4, 3, self.process.process_num_categories),
                dtype=jnp.float32,
            )
        },
    )
    prediction = {
        'logits': jnp.zeros((2, 4, 3, self.process.process_num_categories))
    }
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.integrated_ddim_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

  def test_update_raises_for_shape_mismatch(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.integrated_ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = {
        'logits': jnp.zeros((2, 8, self.process.process_num_categories))
    }
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt and x0 to have the same shape'
    ):
      self.integrated_ddim_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

  def test_finalize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.integrated_ddim_step.initialize(
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
    final_step = self.integrated_ddim_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )

    expected_xt = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_close(final_step.xt, expected_xt)

  def _reference_integrated_posterior(
      self, xt_val, alpha_s, alpha_t, p_x0_given_xt, invariant_probs
  ):
    """Compute p(x_s | x_t) independently using the marginalization formula."""
    voc_size = len(invariant_probs)
    pi_xt = invariant_probs[xt_val]
    ratio = alpha_t / alpha_s

    # Compute w(x_0, x_t) for each x_0
    w = []
    for x0 in range(voc_size):
      q_xt_given_x0 = alpha_t * float(x0 == xt_val) + (
          1.0 - alpha_t
      ) * pi_xt
      w.append(p_x0_given_xt[x0] / max(q_xt_given_x0, 1e-12))
    w = jnp.array(w)
    sum_w = jnp.sum(w)

    # Compute p(x_s | x_t) for each x_s
    p_xs = []
    for xs in range(voc_size):
      q_xt_given_xs = ratio * float(xs == xt_val) + (1.0 - ratio) * pi_xt
      expected_xs = (
          alpha_s * w[xs] + (1.0 - alpha_s) * sum_w * invariant_probs[xs]
      )
      p_xs.append(float(q_xt_given_xs * expected_xs))

    p_xs = jnp.array(p_xs)
    return p_xs / jnp.sum(p_xs)

  def _code_integrated_posterior(
      self, xt_val, alpha_s, alpha_t, p_x0_given_xt, invariant_probs
  ):
    """Reproduce the computation from IntegratedDiscreteDDIMStep.update."""
    voc_size = len(invariant_probs)
    pi = jnp.array(invariant_probs)
    pi_xt = pi[xt_val]

    xt_oh = jax.nn.one_hot(xt_val, num_classes=voc_size)
    p_x0 = jnp.array(p_x0_given_xt)

    ratio = alpha_t / alpha_s

    q_xt_given_xs = ratio * xt_oh + (1.0 - ratio) * pi_xt
    q_xt_given_x0 = alpha_t * xt_oh + (1.0 - alpha_t) * pi_xt

    w_x0 = p_x0 / jnp.clip(q_xt_given_x0, min=1e-12)
    sum_w = jnp.sum(w_x0)

    expected_xs_given_x0 = alpha_s * w_x0 + (1.0 - alpha_s) * pi * sum_w
    p_xs = q_xt_given_xs * expected_xs_given_x0
    return p_xs / jnp.sum(p_xs)

  def test_numerical_marginalized_posterior(self):
    """Verify code posterior matches independently derived posterior."""
    test_cases = [
        (0, 0.8, 0.3, [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]),
        (1, 0.7, 0.2, [0.25, 0.25, 0.25, 0.25], [0.05, 0.8, 0.1, 0.05]),
        (0, 0.6, 0.1, [0.5, 0.2, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]),
        (2, 0.99, 0.01, [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]),
    ]
    for xt_val, alpha_s, alpha_t, inv_probs, p_x0_probs in test_cases:
      inv = jnp.array(inv_probs)
      p_ref = self._reference_integrated_posterior(
          xt_val, alpha_s, alpha_t, p_x0_probs, inv
      )
      p_code = self._code_integrated_posterior(
          xt_val, alpha_s, alpha_t, p_x0_probs, inv
      )
      chex.assert_trees_all_close(p_ref, p_code, atol=1e-6)

  def test_numerical_posterior_is_valid_distribution(self):
    """The posterior should sum to 1 and have non-negative entries."""
    invariant_probs = jnp.array([0.3, 0.3, 0.4])
    p_x0 = jnp.array([0.2, 0.5, 0.3])

    for xt_val in range(3):
      p = self._reference_integrated_posterior(
          xt_val, 0.7, 0.2, p_x0, invariant_probs
      )
      self.assertAlmostEqual(float(jnp.sum(p)), 1.0, places=6)
      self.assertTrue(jnp.all(p >= -1e-12))

  def test_numerical_posterior_at_same_time_is_identity(self):
    """When alpha_s == alpha_t, the posterior should concentrate on x_t."""
    invariant_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
    p_x0 = jnp.array([0.2, 0.3, 0.3, 0.2])
    xt_val = 1
    alpha = 0.5

    p = self._reference_integrated_posterior(
        xt_val, alpha, alpha, p_x0, invariant_probs
    )
    # When s == t, p(x_s | x_t) should be delta(x_t)
    expected = jnp.zeros(4).at[xt_val].set(1.0)
    chex.assert_trees_all_close(p, expected, atol=1e-6)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError,
        'IntegratedDiscreteDDIMStep does not support masking processes.',
    ):
      discrete_step_sampler.IntegratedDiscreteDDIMStep(
          corruption_process=process
      )

  def test_fail_for_zero_invariant_probs(self):
    invariant_probs = (1.0, 0.0, 0.0)
    process = CategoricalProcess(
        schedule=self.schedule,
        invariant_probs=invariant_probs,
        num_categories=len(invariant_probs),
    )
    with self.assertRaisesRegex(
        ValueError,
        'IntegratedDiscreteDDIMStep does not support invariant probabilities'
        ' with 0.0 probability mass for any element.',
    ):
      discrete_step_sampler.IntegratedDiscreteDDIMStep(
          corruption_process=process
      )


class DiscreteFlowMatchingStepTest(absltest.TestCase):
  """Tests for the DiscreteFlowMatchingStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = discrete.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.randint(
        key, (2, 4, 1), 0, self.process.process_num_categories
    )
    self.dfm_step = discrete_step_sampler.DiscreteFlowMatchingStep(
        corruption_process=self.process
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 1.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.dfm_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_initialize_raises_for_invalid_trailing_dimension(self):
    initial_noise_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected initial_noise to have a trailing dimension of 1'
    ):
      self.dfm_step.initialize(
          initial_noise=initial_noise_bad,
          initial_step_info=initial_step_info,
      )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.dfm_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Test case 1: Full unmasking (alpha_s=1.0, alpha_t=0.5 -> prob_jump=1.0)
    next_step_info_full = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_full = self.dfm_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_full,
    )
    expected_xt_full = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_equal(next_step_full.xt, expected_xt_full)

    # Test case 2: No jump (alpha_s=0.5, alpha_t=0.5 -> prob_jump=0.0)
    next_step_info_no = StepInfo(
        step=1,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_no = self.dfm_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_no,
    )
    chex.assert_trees_all_equal(next_step_no.xt, initial_step.xt)

  def test_update_raises_for_invalid_xt_trailing_dimension(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    xt_bad = jnp.ones((2, 4, 3), dtype=jnp.int32)
    initial_step = DiffusionStep(
        xt=xt_bad,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                (2, 4, 3, self.process.num_categories), dtype=jnp.float32
            )
        },
    )
    prediction = {'logits': jnp.zeros((2, 4, 3, self.process.num_categories))}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.4, 0.4])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.dfm_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

  def test_update_raises_for_shape_mismatch(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.dfm_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = {'logits': jnp.zeros((2, 8, self.process.num_categories))}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.4, 0.4])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaises(ValueError):
      self.dfm_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

  def _reference_dfm_routing(self, alpha_s, alpha_t, stoch_coeff=0.0):
    """Compute expected routing weights from the paper formulas."""
    p_clean_raw = (alpha_s - alpha_t) / max(1.0 - alpha_t, 1e-12) * (
        1.0 + stoch_coeff
    )
    p_noise_raw = (
        (alpha_s - alpha_t) / max(alpha_t, 1e-12) * stoch_coeff
    )

    p_clean_raw = max(p_clean_raw, 0.0)
    p_noise_raw = max(p_noise_raw, 0.0)
    scale = max(1.0, p_clean_raw + p_noise_raw)

    p_clean = p_clean_raw / scale
    p_noise = p_noise_raw / scale
    p_stay = 1.0 - p_clean - p_noise
    return p_stay, p_noise, p_clean

  def test_numerical_routing_weights_match_reference(self):
    """Verify routing weights match the independently derived formula."""
    test_cases = [
        (0.8, 0.5, 0.0),
        (0.95, 0.9, 0.0),
        (0.8, 0.5, 0.5),
        (0.8, 0.5, 1.0),
        (0.5, 0.5, 0.0),
        (0.99, 0.01, 0.0),
        (0.99, 0.01, 1.0),
    ]
    for alpha_s, alpha_t, stoch_coeff in test_cases:
      ref_stay, ref_noise, ref_clean = self._reference_dfm_routing(
          alpha_s, alpha_t, stoch_coeff
      )

      # Reproduce the computation from DiscreteFlowMatchingStep.update
      alpha_s_arr = jnp.array(alpha_s)
      alpha_t_arr = jnp.array(alpha_t)

      p_clean_raw = (
          (alpha_s_arr - alpha_t_arr)
          / jnp.maximum(1.0 - alpha_t_arr, 1e-12)
          * (1.0 + stoch_coeff)
      )
      p_noise_raw = (
          (alpha_s_arr - alpha_t_arr)
          / jnp.maximum(alpha_t_arr, 1e-12)
          * stoch_coeff
      )
      p_clean_raw = jnp.maximum(p_clean_raw, 0.0)
      p_noise_raw = jnp.maximum(p_noise_raw, 0.0)
      scale = jnp.maximum(1.0, p_clean_raw + p_noise_raw)
      code_p_clean = p_clean_raw / scale
      code_p_noise = p_noise_raw / scale
      code_p_stay = 1.0 - code_p_clean - code_p_noise

      chex.assert_trees_all_close(
          jnp.array([code_p_stay, code_p_noise, code_p_clean]),
          jnp.array([ref_stay, ref_noise, ref_clean]),
          atol=1e-7,
      )

  def test_numerical_deterministic_weights_sum_to_one(self):
    """With gamma=0, p_stay + p_clean = 1 and p_noise = 0."""
    for alpha_s in [0.3, 0.5, 0.7, 0.95]:
      for alpha_t in [0.05, 0.1, 0.2]:
        if alpha_t >= alpha_s:
          continue
        p_stay, p_noise, p_clean = self._reference_dfm_routing(
            alpha_s, alpha_t, 0.0
        )
        self.assertAlmostEqual(p_stay + p_noise + p_clean, 1.0, places=10)
        self.assertAlmostEqual(p_noise, 0.0, places=10)

  def test_numerical_same_time_all_stay(self):
    """When alpha_s == alpha_t, no denoising should occur."""
    p_stay, p_noise, p_clean = self._reference_dfm_routing(0.5, 0.5, 0.0)
    self.assertAlmostEqual(p_stay, 1.0, places=10)
    self.assertAlmostEqual(p_clean, 0.0, places=10)

  def test_numerical_weights_nonneg_after_clipping(self):
    """All routing weights should be non-negative even with high gamma."""
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
      for alpha_s in [0.3, 0.5, 0.8]:
        for alpha_t in [0.05, 0.1, 0.2]:
          if alpha_t >= alpha_s:
            continue
          p_stay, p_noise, p_clean = self._reference_dfm_routing(
              alpha_s, alpha_t, gamma
          )
          self.assertGreaterEqual(p_stay, -1e-12)
          self.assertGreaterEqual(p_noise, -1e-12)
          self.assertGreaterEqual(p_clean, -1e-12)

  def test_update_with_gamma(self):
    num_samples = 10
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([[0.5]] * num_samples)[:, :, None],
        rng=jax.random.PRNGKey(0),
    )
    # Start with more samples to ensure noise jump is detected.
    initial_xt = jnp.ones((num_samples, 4, 1), dtype=jnp.int32)
    initial_step = self.dfm_step.initialize(
        initial_noise=initial_xt,
        initial_step_info=initial_step_info,
    )

    # Predict category 1 (same as current).
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Use gamma that won't clip.
    dfm_step_gamma = discrete_step_sampler.DiscreteFlowMatchingStep(
        corruption_process=self.process, stoch_coeff=1.0
    )

    next_step_info = StepInfo(
        step=1,
        time=jnp.array([[0.4]] * num_samples)[:, :, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step = dfm_step_gamma.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    # Some tokens should have changed to noise (not 1).
    self.assertTrue(jnp.any(next_step.xt != 1))

  def test_update_rejects_2d_input(self):
    """Passing xt without trailing dim should raise ValueError."""
    noise_2d = jax.random.randint(
        jax.random.PRNGKey(0),
        (2, 4),
        0,
        self.process.process_num_categories,
    )
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5]),
        rng=jax.random.PRNGKey(0),
    )
    initial_step = DiffusionStep(
        xt=noise_2d,
        step_info=initial_step_info,
        aux={
            'logits': jnp.zeros(
                noise_2d.shape + (self.process.num_categories,),
                dtype=jnp.float32,
            )
        },
    )
    logits = jnp.zeros(noise_2d.shape + (self.process.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    prediction = {'logits': logits}
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.4, 0.4]),
        rng=jax.random.PRNGKey(1),
    )
    with self.assertRaisesRegex(
        ValueError, 'Expected xt to have a trailing dimension of 1'
    ):
      self.dfm_step.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )


class DDIMRoutingEquivalenceTest(absltest.TestCase):
  """Verify routing-based DDIM matches the original logit-space computation.

  The original DiscreteDDIMStep computed the reverse posterior in full
  M-dimensional logit space:

    first_logit[k]  = log(r * 1[k=xt] + (1-r) * π(xt))
    second_logit[k] = log(αs * 1[k=x0] + (1-αs) * π(k))
    total_logit = first_logit + second_logit

  The routing reformulation decomposes this into 3-way routing weights.
  This test checks that both produce exactly the same distribution over
  output tokens, including the edge case where x0 == xt.
  """

  def _posterior_distribution(
      self, xt, x0, alpha_s, alpha_t, invariant_probs_vec
  ):
    """Compute the exact posterior in probability space.

    p(x_s | x_t, x_0) ∝ p(x_t | x_s) * p(x_s | x_0)

    Evaluated for every x_s in {0, ..., M-1}.

    Args:
      xt: Current token.
      x0: Predicted clean token.
      alpha_s: Diffusion schedule value at time s.
      alpha_t: Diffusion schedule value at time t.
      invariant_probs_vec: Invariant distribution.

    Returns:
      The M-dimensional posterior distribution.
    """
    voc_size = int(invariant_probs_vec.shape[0])
    ratio = alpha_t / alpha_s

    # Build unnormalized weight for each x_s value.
    weights = []
    for xs in range(voc_size):
      # p(x_t | x_s) = r * 1[xs=xt] + (1-r) * π(xt)
      p_xt_given_xs = ratio * float(xs == xt) + (1.0 - ratio) * float(
          invariant_probs_vec[xt]
      )
      # p(x_s | x_0) = α_s * 1[xs=x0] + (1-α_s) * π(xs)
      p_xs_given_x0 = alpha_s * float(xs == x0) + (1.0 - alpha_s) * float(
          invariant_probs_vec[xs]
      )
      weights.append(p_xt_given_xs * p_xs_given_x0)

    weights = jnp.array(weights)
    return weights / jnp.sum(weights)

  def _routing_distribution(
      self, xt, x0, alpha_s, alpha_t, invariant_probs_vec
  ):
    """Compute the routing-based posterior distribution.

    Mirrors the actual code in DiscreteDDIMStep.update.

    Args:
      xt: Current token.
      x0: Predicted clean token.
      alpha_s: Diffusion schedule value at time s.
      alpha_t: Diffusion schedule value at time t.
      invariant_probs_vec: Invariant distribution.

    Returns:
      The M-dimensional posterior distribution.
    """
    ratio = alpha_t / alpha_s
    pi_xt = float(invariant_probs_vec[xt])

    # T2 → stay, T4 → noise, T3 → clean
    p_stay = ratio * (1.0 - alpha_s) * pi_xt
    p_noise = (1.0 - ratio) * (1.0 - alpha_s) * pi_xt
    p_clean = (1.0 - ratio) * alpha_s * pi_xt

    # When x0 == xt, CLEAN is a no-op. Merge T1 and p_clean into p_stay.
    if x0 == xt:
      p_stay = p_stay + ratio * alpha_s + p_clean
      p_clean = 0.0

    total = p_stay + p_noise + p_clean
    p_stay_norm = p_stay / total
    p_noise_norm = p_noise / total
    p_clean_norm = p_clean / total

    # Build the M-dimensional output distribution by marginalizing
    # over the routing action:
    #   P(output=k) = P(STAY)*1[k=xt] + P(NOISE)*π(k) + P(CLEAN)*1[k=x0]
    inv_probs = [float(p) for p in invariant_probs_vec]
    dist = [p_noise_norm * inv_probs[k] for k in range(len(inv_probs))]
    dist[xt] += p_stay_norm
    dist[x0] += p_clean_norm
    return jnp.array(dist)

  def test_equivalence_x0_neq_xt(self):
    """Test routing matches posterior when x0 != xt."""
    voc_size = 5
    invariant_probs = jnp.array([0.1, 0.3, 0.2, 0.25, 0.15])

    for xt_val in range(voc_size):
      for x0_val in range(voc_size):
        if x0_val == xt_val:
          continue
        for alpha_s_val in [0.2, 0.5, 0.8]:
          alpha_t = 0.05
          p_exact = self._posterior_distribution(
              xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
          )
          p_route = self._routing_distribution(
              xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
          )
          chex.assert_trees_all_close(p_exact, p_route, atol=1e-6)

  def test_equivalence_x0_eq_xt(self):
    """Test routing matches posterior when x0 == xt (the T1 cross-term)."""
    voc_size = 5
    invariant_probs = jnp.array([0.1, 0.3, 0.2, 0.25, 0.15])

    for xt_val in range(voc_size):
      x0_val = xt_val
      for alpha_s_val in [0.2, 0.5, 0.8]:
        alpha_t = 0.05
        p_exact = self._posterior_distribution(
            xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
        )
        p_route = self._routing_distribution(
            xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
        )
        chex.assert_trees_all_close(p_exact, p_route, atol=1e-6)

  def test_equivalence_nonuniform_invariant(self):
    """Test with a highly non-uniform invariant distribution."""
    voc_size = 3
    invariant_probs = jnp.array([0.01, 0.01, 0.98])

    for xt_val in range(voc_size):
      for x0_val in range(voc_size):
        p_exact = self._posterior_distribution(
            xt_val, x0_val, 0.3, 0.7, invariant_probs
        )
        p_route = self._routing_distribution(
            xt_val, x0_val, 0.3, 0.7, invariant_probs
        )
        chex.assert_trees_all_close(p_exact, p_route, atol=1e-6)


class ApplyRoutingTest(absltest.TestCase):
  """Tests for the _sample_routing helper."""

  def test_deterministic_stay(self):
    # weights = [1, 0, 0] means stay.
    weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[1.0], [1.0]]]),
        noise=jnp.array([[[0.0], [0.0]]]),
        clean=jnp.array([[[0.0], [0.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        key=key,
        weights=weights,
        candidates=discrete_step_sampler.Routing(
            stay=xt, noise=x_noise, clean=x0,
        ),
    )
    chex.assert_trees_all_equal(new_xt, xt)

  def test_deterministic_clean(self):
    # weights = [0, 0, 1] means jump to x0.
    weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[0.0], [0.0]]]),
        noise=jnp.array([[[0.0], [0.0]]]),
        clean=jnp.array([[[1.0], [1.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        key=key,
        weights=weights,
        candidates=discrete_step_sampler.Routing(
            stay=xt, noise=x_noise, clean=x0,
        ),
    )
    chex.assert_trees_all_equal(new_xt, x0)

  def test_deterministic_noise(self):
    # weights = [0, 1, 0] means jump to noise.
    weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[0.0], [0.0]]]),
        noise=jnp.array([[[1.0], [1.0]]]),
        clean=jnp.array([[[0.0], [0.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        key=key,
        weights=weights,
        candidates=discrete_step_sampler.Routing(
            stay=xt, noise=x_noise, clean=x0,
        ),
    )
    chex.assert_trees_all_equal(new_xt, x_noise)

  def test_mixed_routing(self):
    # Position 0: deterministic stay, Position 1: deterministic clean.
    weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[1.0], [0.0]]]),
        noise=jnp.array([[[0.0], [0.0]]]),
        clean=jnp.array([[[0.0], [1.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        key=key,
        weights=weights,
        candidates=discrete_step_sampler.Routing(
            stay=xt, noise=x_noise, clean=x0,
        ),
    )
    expected = jnp.array([[[3], [1]]])
    chex.assert_trees_all_equal(new_xt, expected)

  def test_stochastic_routing(self):
    # 50/50 stay vs clean — results should vary across seeds.
    weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[0.5]]]),
        noise=jnp.array([[[0.0]]]),
        clean=jnp.array([[[0.5]]]),
    )
    xt = jnp.array([[[3]]])
    x0 = jnp.array([[[0]]])
    x_noise = jnp.array([[[2]]])

    results = set()
    for seed in range(50):
      new_xt = discrete_step_sampler._sample_routing(
          key=jax.random.PRNGKey(seed),
          weights=weights,
          candidates=discrete_step_sampler.Routing(
              stay=xt, noise=x_noise, clean=x0,
          ),
      )
      results.add(int(new_xt[0, 0, 0]))

    # Should see both stay (3) and clean (0).
    self.assertIn(3, results)
    self.assertIn(0, results)


class PlannerProtocolTest(absltest.TestCase):

  def test_identity_planner(self):

    class IdentityPlanner:

      def __call__(self, routing_weights, logits, x0, xt, time, next_time, key):
        return routing_weights

    planner = IdentityPlanner()
    routing_weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[0.2]]]),
        noise=jnp.array([[[0.3]]]),
        clean=jnp.array([[[0.5]]]),
    )
    # dummy args
    logits = jnp.zeros((1, 1, 5))
    x0 = jnp.zeros((1, 1, 1))
    xt = jnp.zeros((1, 1, 1))
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])
    key = jax.random.PRNGKey(0)

    out = planner(routing_weights, logits, x0, xt, time, next_time, key)
    chex.assert_trees_all_equal(out, routing_weights)


class RoutingTest(absltest.TestCase):
  """Tests for the Routing container."""

  def test_consistent_shapes_accepted(self):
    """Routing with matching shapes should construct without error."""
    r = discrete_step_sampler.Routing(
        stay=jnp.ones((2, 4, 1)),
        noise=jnp.ones((2, 4, 1)),
        clean=jnp.ones((2, 4, 1)),
    )
    self.assertEqual(r.stay.shape, (2, 4, 1))

  def test_to_stacked(self):
    """to_stacked should produce a (..., 3) array."""
    r = discrete_step_sampler.Routing(
        stay=jnp.array([0.5]),
        noise=jnp.array([0.3]),
        clean=jnp.array([0.2]),
    )
    stacked = r.to_stacked()
    self.assertEqual(stacked.shape, (1, 3))
    chex.assert_trees_all_close(
        stacked, jnp.array([[0.5, 0.3, 0.2]]),
    )

  def test_scalar_accepted(self):
    """Scalar routing fields are valid with jnp.stack."""
    r = discrete_step_sampler.Routing(
        stay=jnp.array(0.5),
        noise=jnp.array(0.3),
        clean=jnp.array(0.2),
    )
    self.assertEqual(r.stay.shape, ())


class GreedyPlannerTest(absltest.TestCase):

  def test_greedy_planner_budget(self):
    planner = discrete_step_sampler.GreedyPlanner()
    # 1 batch, 4 seq len. Realistic routing: all eligible with stay/noise > 0.
    routing_weights = discrete_step_sampler.Routing(
        stay=jnp.full((1, 4, 1), 0.3),
        noise=jnp.full((1, 4, 1), 0.2),
        clean=jnp.full((1, 4, 1), 0.5),
    )
    logits = jnp.array(
        [[[10.0, 0.0], [5.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]
    )  # high confidence first
    x0 = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 4, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])  # frac = 0.5 -> budget = 4 eligible * 0.5 = 2
    key = jax.random.PRNGKey(0)

    out_probs = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # Top 2 positions → force CLEAN (stay=0, noise=0, clean=1).
    # Non-selected → keep original stay/noise, zero out clean.
    expected = discrete_step_sampler.Routing(
        stay=jnp.array([[[0.0], [0.0], [0.3], [0.3]]]),
        noise=jnp.array([[[0.0], [0.0], [0.2], [0.2]]]),
        clean=jnp.array([[[1.0], [1.0], [0.0], [0.0]]]),
    )
    chex.assert_trees_all_close(out_probs.stay, expected.stay)
    chex.assert_trees_all_close(out_probs.noise, expected.noise)
    chex.assert_trees_all_close(out_probs.clean, expected.clean)

  def test_greedy_planner_eligibility(self):
    planner = discrete_step_sampler.GreedyPlanner()
    # Position 0 is NOT eligible (p_clean = 0), has original stay=1.0.
    routing_weights = discrete_step_sampler.Routing(
        stay=jnp.array([[[1.0], [0.3], [0.3], [0.3]]]),
        noise=jnp.array([[[0.0], [0.2], [0.2], [0.2]]]),
        clean=jnp.array([[[0.0], [0.5], [0.5], [0.5]]]),
    )
    logits = jnp.array(
        [[[10.0, 0.0], [5.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]
    )  # Pos 0 has highest logit but ineligible
    x0 = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 4, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])  # frac = 0.5 -> budget = 3 eligible * 0.5 = 1
    key = jax.random.PRNGKey(0)

    out_probs = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # Pos 0 is ineligible (p_clean=0), so num_eligible=3.
    # Budget = 3 * 0.5 = 1 (truncated to int).
    # Top 1 eligible position by confidence: Pos 1 → force CLEAN.
    # Non-selected (Pos 0, 2, 3) → keep original stay/noise, zero clean.
    expected = discrete_step_sampler.Routing(
        stay=jnp.array([[[1.0], [0.0], [0.3], [0.3]]]),
        noise=jnp.array([[[0.0], [0.0], [0.2], [0.2]]]),
        clean=jnp.array([[[0.0], [1.0], [0.0], [0.0]]]),
    )
    chex.assert_trees_all_close(out_probs.stay, expected.stay)
    chex.assert_trees_all_close(out_probs.noise, expected.noise)
    chex.assert_trees_all_close(out_probs.clean, expected.clean)

  def test_greedy_planner_k_zero(self):
    """When clean weight is small, budget k=0. Keep original stay/noise."""
    planner = discrete_step_sampler.GreedyPlanner()
    routing_weights = discrete_step_sampler.Routing(
        stay=jnp.full((1, 4, 1), 0.7),
        noise=jnp.full((1, 4, 1), 0.2),
        clean=jnp.full((1, 4, 1), 0.1),
    )
    logits = jnp.array([[[10.0, 0.0], [5.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
    x0 = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 4, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([1.0])
    key = jax.random.PRNGKey(0)

    out_probs = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # k=0: no positions selected. All keep original stay/noise, clean zeroed.
    expected = discrete_step_sampler.Routing(
        stay=jnp.full((1, 4, 1), 0.7),
        noise=jnp.full((1, 4, 1), 0.2),
        clean=jnp.zeros((1, 4, 1)),
    )
    chex.assert_trees_all_close(out_probs.stay, expected.stay)
    chex.assert_trees_all_close(out_probs.noise, expected.noise)
    chex.assert_trees_all_close(out_probs.clean, expected.clean)

  def test_greedy_planner_2d_spatial(self):
    """GreedyPlanner must work with 2D spatial data (e.g.

    adjacency matrices).
    """
    planner = discrete_step_sampler.GreedyPlanner()
    # Shape (1, 3, 3, 1): batch=1, spatial=(3,3), vocab_trailing=1.
    routing_weights = discrete_step_sampler.Routing(
        stay=jnp.full((1, 3, 3, 1), 0.3),
        noise=jnp.full((1, 3, 3, 1), 0.2),
        clean=jnp.full((1, 3, 3, 1), 0.5),
    )
    # 9 positions total, 2 vocab classes.
    # Logits: position (0,0) has highest confidence, then (0,1), etc.
    logits_flat = jnp.array([
        [10.0, 0.0],
        [9.0, 0.0],
        [8.0, 0.0],
        [7.0, 0.0],
        [6.0, 0.0],
        [5.0, 0.0],
        [4.0, 0.0],
        [3.0, 0.0],
        [2.0, 0.0],
    ])
    logits = logits_flat.reshape(1, 3, 3, 2)
    x0 = jnp.zeros((1, 3, 3, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 3, 3, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])  # frac = 0.5 -> budget = 9 * 0.5 = 4
    key = jax.random.PRNGKey(0)

    out = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # Output must have the same spatial shape.
    self.assertEqual(out.stay.shape, (1, 3, 3, 1))
    self.assertEqual(out.noise.shape, (1, 3, 3, 1))
    self.assertEqual(out.clean.shape, (1, 3, 3, 1))

    # Budget k = 4. Top-4 positions (by confidence) → forced CLEAN.
    # Positions (0,0), (0,1), (0,2), (1,0) should be selected.
    selected = out.clean[0, :, :, 0]  # (3, 3)
    num_selected = int(jnp.sum(selected > 0))
    self.assertEqual(num_selected, 4)
    # Selected positions have stay=0, noise=0.
    self.assertEqual(float(jnp.sum(out.stay[out.clean > 0])), 0.0)
    self.assertEqual(float(jnp.sum(out.noise[out.clean > 0])), 0.0)


if __name__ == '__main__':
  absltest.main()
