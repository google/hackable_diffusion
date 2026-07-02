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

"""Tests for sampling."""

import dataclasses
from typing import Callable

import chex
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import diffusion_early_stopping
from hackable_diffusion.lib.sampling import sampling
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Type Aliases
################################################################################

SamplerStep = base.SamplerStep

################################################################################
# MARK: Helper Functions
################################################################################

dummy_inference_fn = lambda xt, conditioning, time: {'x0': xt}


shift_right = lambda x: jnp.roll(x, 1, axis=-1)
invert = lambda x: 1.0 - x


@dataclasses.dataclass(frozen=True, kw_only=True)
class DummyStep(SamplerStep):

  def initialize(self, initial_noise, initial_step_info):
    return base.DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  def update(self, prediction, current_step, next_step_info):
    return base.DiffusionStep(
        xt=shift_right(prediction['x0']),
        step_info=next_step_info,
        aux=dict(),
    )

  def finalize(self, prediction, current_step, next_step_info):
    return base.DiffusionStep(
        xt=invert(prediction['x0']),
        step_info=next_step_info,
        aux=dict(),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DummyStepWithLogits(SamplerStep):
  """Like DummyStep but populates aux['logits'] via a callable.

  Attributes:
    logits_fn: A callable ``(xt) -> logits`` that computes logits from the
      current xt.  The returned logits are stored in ``aux['logits']``.
  """

  logits_fn: Callable[..., jax.Array]

  def initialize(self, initial_noise, initial_step_info):
    return base.DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': self.logits_fn(initial_noise)},
    )

  def update(self, prediction, current_step, next_step_info):
    new_xt = shift_right(prediction['x0'])
    return base.DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': self.logits_fn(new_xt)},
    )

  def finalize(self, prediction, current_step, next_step_info):
    new_xt = invert(prediction['x0'])
    return base.DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': self.logits_fn(new_xt)},
    )


################################################################################
# MARK: Early Stopping Fakes
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class StopAfterNSteps:
  """Early stopping that is trigerred after ``num_steps`` update steps."""

  num_steps: int

  def should_stop(self, *, step, current_step, previous_step):
    del current_step, previous_step
    batch_size = 2  # Matches test setup.
    return jnp.full((batch_size,), step >= self.num_steps - 1, dtype=jnp.bool_)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PerElementStop:
  """Early stopping that is trigerred per batch element at different steps."""

  stop_at_step: tuple[int, ...]

  def should_stop(self, *, step, current_step, previous_step):
    del current_step, previous_step
    return step >= jnp.array(self.stop_at_step)


@dataclasses.dataclass(frozen=True)
class AlwaysStop:
  """Immediately requests stop for all elements on every call."""

  def should_stop(self, *, step, current_step, previous_step):
    del step, previous_step
    batch_size = current_step.xt.shape[0]
    return jnp.ones(batch_size, dtype=jnp.bool_)


################################################################################
# MARK: DiffusionSampler Tests (scan-based)
################################################################################


class DiffusionSamplerTest(parameterized.TestCase):
  """Tests for the scan-based DiffusionSampler."""

  def setUp(self):
    super().setUp()
    self.time_schedule = time_scheduling.UniformTimeSchedule()
    self.stepper = DummyStep()
    self.initial_noise = jnp.repeat(
        jnp.expand_dims(jnp.eye(4), axis=0), 2, axis=0
    )
    self.conditioning = dict()
    self.dummy_inference_fn = dummy_inference_fn

  # MARK: Test for Helper Functions

  def test_split_pytree(self):
    first, intermediates, last = sampling._split_pytree(
        dict(
            a=jnp.array([1, 2, 3, 4]),
            b=jnp.array([5, 6, 7, 8]),
        )
    )

    chex.assert_trees_all_equal(first, dict(a=1, b=5))
    chex.assert_trees_all_equal(
        intermediates,
        dict(
            a=jnp.array([2, 3]),
            b=jnp.array([6, 7]),
        ),
    )
    chex.assert_trees_all_equal(last, dict(a=4, b=8))

  def test_concat_pytree(self):
    first = dict(a=1, b=5)
    intermediates = dict(
        a=jnp.array([2, 3]),
        b=jnp.array([6, 7]),
    )
    last = dict(a=4, b=8)

    chex.assert_trees_all_equal(
        sampling._concat_pytree(first, intermediates, last),
        dict(
            a=jnp.array([1, 2, 3, 4]),
            b=jnp.array([5, 6, 7, 8]),
        ),
    )

  # MARK: Test for diffusion_sampling

  def test_sample_one(self):
    """Test the sampling function on a toy example."""

    sample_fn = sampling.DiffusionSampler(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
    )
    last_step, all_steps = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # confirm that all steps have the correct xt
    all_xt = all_steps.xt
    chex.assert_trees_all_equal(
        all_xt,
        jnp.repeat(
            jnp.array([
                [  # step 0 - init
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ],
                [  # step 1 - shift right
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                ],
                [  # step 2 - shift right
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                ],
                [  # step 3 - shift right
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                ],
                [  # step 4 - invert
                    [
                        [1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0],
                    ],
                ],
            ]),
            repeats=2,
            axis=1,
        ),
    )

    # confirm that the last step is the same as the carry
    chex.assert_trees_all_equal(all_xt[-1], last_step.xt)

  @parameterized.named_parameters(
      ('zero_steps', 0),
      ('negative_steps', -1),
      ('one_step', 1),
  )
  def test_raises_error_for_less_than_two_steps(self, num_steps: int):
    """Tests that an error is raised for a non-positive number of steps."""
    sample_fn = sampling.DiffusionSampler(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=num_steps,
    )

    with self.assertRaisesRegex(
        ValueError, 'Number of steps must be at least 2.*'
    ):
      sample_fn(
          inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
          initial_noise=self.initial_noise,
          conditioning=self.conditioning,
          rng=jax.random.PRNGKey(0),
      )

  def test_sample_one_2_steps(self):
    """Test the sampling function on a toy example.

    We only run 2 steps to make sure that the scan is omitted in that case.
    """

    sample_fn = sampling.DiffusionSampler(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=2,
    )
    last_step, all_steps = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # confirm that all steps have the correct xt
    all_xt = all_steps.xt
    chex.assert_trees_all_equal(
        all_xt,
        jnp.repeat(
            jnp.array([
                [  # step 0 - init
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ],
                [  # step 1 - invert
                    [
                        [0.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0, 0.0],
                    ],
                ],
            ]),
            repeats=2,
            axis=1,
        ),
    )

    # confirm that the last step is the same as the carry
    chex.assert_trees_all_equal(all_xt[-1], last_step.xt)


################################################################################
# MARK: DiffusionSamplerWithEarlyStopping Tests (while_loop-based)
################################################################################


class DiffusionSamplerWithEarlyStoppingTest(parameterized.TestCase):
  """Tests for the while_loop-based DiffusionSamplerWithEarlyStopping."""

  def setUp(self):
    super().setUp()
    self.time_schedule = time_scheduling.UniformTimeSchedule()
    self.stepper = DummyStep()
    self.initial_noise = jnp.repeat(
        jnp.expand_dims(jnp.eye(4), axis=0), 2, axis=0
    )
    self.conditioning = dict()
    self.dummy_inference_fn = dummy_inference_fn

  # MARK: Basic sampling

  def test_sample_5_steps_last_step(self):
    """Test that the final step matches the expected result with 5 steps.

    With num_steps=5 the while_loop runs 4 update steps (shift_right).
    shift_right^4 on a 4x4 identity wraps back to identity, then
    finalize=invert gives 1-eye(4).
    """
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
    )
    last_step, trajectory = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    self.assertIsNone(trajectory)

    # 4 shift_rights on 4x4 eye = eye, then invert = 1-eye.
    expected_last = jnp.repeat(
        jnp.expand_dims(
            jnp.array([
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]),
            axis=0,
        ),
        repeats=2,
        axis=0,
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)

  @parameterized.named_parameters(
      ('zero_steps', 0),
      ('negative_steps', -1),
  )
  def test_raises_error_for_less_than_one_step(self, num_steps: int):
    """Tests that an error is raised for a non-positive number of steps."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=num_steps,
    )

    with self.assertRaisesRegex(
        ValueError, 'Number of steps must be at least 1.*'
    ):
      sample_fn(
          inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
          initial_noise=self.initial_noise,
          conditioning=self.conditioning,
          rng=jax.random.PRNGKey(0),
      )

  def test_sample_1_step(self):
    """With 1 step: init + finalize only (no update steps)."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=1,
    )
    last_step, trajectory = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    self.assertIsNone(trajectory)

    expected_last = jnp.repeat(
        jnp.expand_dims(
            jnp.array([
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]),
            axis=0,
        ),
        repeats=2,
        axis=0,
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)

  def test_sample_2_steps(self):
    """With 2 steps: init -> 1 update -> finalize."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=2,
    )
    last_step, trajectory = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    self.assertIsNone(trajectory)

    # init=identity, 1 shift_right, then finalize=invert.
    shifted = shift_right(jnp.eye(4))
    expected_single = invert(shifted)
    expected_last = jnp.repeat(
        jnp.expand_dims(expected_single, axis=0),
        repeats=2,
        axis=0,
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)

  def test_sample_3_steps(self):
    """With 3 steps: init -> 2 shift_rights -> finalize."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=3,
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    shifted = shift_right(shift_right(jnp.eye(4)))
    expected_single = invert(shifted)
    expected_last = jnp.repeat(
        jnp.expand_dims(expected_single, axis=0),
        repeats=2,
        axis=0,
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)

  def test_output_shape(self):
    """Output shape matches input shape."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )
    self.assertEqual(last_step.xt.shape, self.initial_noise.shape)

  # MARK: Step counter

  def test_step_counter_without_early_stopping(self):
    """step_info.step records actual total steps (updates + finalize)."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )
    # 5 steps: 4 updates + 1 finalize = 5, but step counter = updates + 1
    # num_steps=5 → 4 intermediate update steps + 1 finalize = 5
    self.assertEqual(int(last_step.step_info.step), 5)

  def test_step_counter_1_step(self):
    """With 1 step: 0 updates + 1 finalize = 1."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=1,
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )
    # num_steps=1 → 0 update steps + 1 finalize = 1
    self.assertEqual(int(last_step.step_info.step), 1)

  # MARK: Early stopping

  def test_early_stop_after_1_update(self):
    """Early stopping after 1 update step (out of 4 possible)."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
        early_stopping_fn=StopAfterNSteps(num_steps=1),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # init=identity, 1 shift_right, then finalize=invert.
    shifted = shift_right(jnp.eye(4))
    expected_single = invert(shifted)
    expected_last = jnp.repeat(
        jnp.expand_dims(expected_single, axis=0), repeats=2, axis=0
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)

    # Step counter: 1 update + 1 (finalize) = 2
    self.assertEqual(int(last_step.step_info.step), 2)

  def test_always_stop_skips_all_updates(self):
    """AlwaysStop fires on the first update, so no updates are effective.

    The body runs once (for step=0), but done is immediately set to True,
    so the loop exits after that single iteration.
    """
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
        early_stopping_fn=AlwaysStop(),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # The body executes once (step=0): shift_right is computed but then
    # early_stop fires.  Since done was False before this iteration,
    # the update IS applied (freeze only protects already-done elements).
    # So: init=identity -> 1 shift_right -> finalize=invert.
    shifted = shift_right(jnp.eye(4))
    expected_single = invert(shifted)
    expected_last = jnp.repeat(
        jnp.expand_dims(expected_single, axis=0), repeats=2, axis=0
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)

    # 1 update step + 1 = 2
    self.assertEqual(int(last_step.step_info.step), 2)

  def test_no_early_stop_runs_all_steps(self):
    """NoEarlyStop (default) runs all update steps."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
        early_stopping_fn=diffusion_early_stopping.DiffusionNoEarlyStopFn(),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )
    # Same as test_sample_5_steps_last_step: 4 shift_rights + invert.
    expected_last = jnp.repeat(
        jnp.expand_dims(
            jnp.array([
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]),
            axis=0,
        ),
        repeats=2,
        axis=0,
    )
    chex.assert_trees_all_equal(last_step.xt, expected_last)
    self.assertEqual(int(last_step.step_info.step), 5)

  def test_per_element_early_stop(self):
    """Different batch elements stop at different update steps.

    Element 0 stops after update step 0, element 1 stops after step 2.
    The while_loop continues until both are done (step 2).
    Element 0's carry is frozen after step 0.
    """
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
        early_stopping_fn=PerElementStop(stop_at_step=(0, 2)),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # Element 0: 1 shift_right (frozen after step 0), then finalize=invert.
    elem0_shifted = shift_right(jnp.eye(4))
    elem0_expected = invert(elem0_shifted)

    # Element 1: 3 shift_rights (steps 0,1,2), then finalize=invert.
    elem1_shifted = jnp.eye(4)
    for _ in range(3):
      elem1_shifted = shift_right(elem1_shifted)
    elem1_expected = invert(elem1_shifted)

    expected = jnp.stack([elem0_expected, elem1_expected], axis=0)
    chex.assert_trees_all_equal(last_step.xt, expected)

  def test_step_counter_with_early_stop(self):
    """Step counter reflects actual steps when early stopping fires early."""
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=10,
        early_stopping_fn=StopAfterNSteps(num_steps=2),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )
    # 2 update steps + 1 (finalize) = 3
    self.assertEqual(int(last_step.step_info.step), 3)

  # MARK: Equivalence with scan-based sampler

  @parameterized.named_parameters(
      ('2_steps', 2),
      ('3_steps', 3),
      ('5_steps', 5),
      ('10_steps', 10),
  )
  def test_equivalence_with_scan_sampler(self, num_steps: int):
    """The while_loop sampler produces the same xt as the scan sampler.

    The scan-based DiffusionSampler with ``num_steps=N`` performs ``N-2``
    update steps, while DiffusionSamplerWithEarlyStopping with
    ``num_steps=M`` performs ``M-1`` update steps.  To get the same number
    of updates we set ``M = N - 1``.

    Because the dummy inference function ignores time, the different time
    schedules do not affect the result.

    Args:
      num_steps: Number of diffusion steps for the scan-based sampler.
    """
    rng = jax.random.PRNGKey(42)

    scan_sampler = sampling.DiffusionSampler(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=num_steps,
        store_trajectory=False,
    )
    # N-2 updates (scan) == M-1 updates (while_loop) ⇒ M = N-1.
    whileloop_sampler = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=num_steps - 1,
    )

    scan_result, _ = scan_sampler(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=rng,
    )
    whileloop_result, _ = whileloop_sampler(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=rng,
    )

    chex.assert_trees_all_close(scan_result.xt, whileloop_result.xt)


################################################################################
# MARK: EntropyEarlyStopIntegrationTest
################################################################################


class EntropyEarlyStopIntegrationTest(parameterized.TestCase):
  """Integration tests: DiffusionEntropyEarlyStopFn + DiffusionSamplerWithEarlyStopping."""

  def setUp(self):
    super().setUp()
    self.time_schedule = time_scheduling.UniformTimeSchedule()
    self.initial_noise = jnp.repeat(
        jnp.expand_dims(jnp.eye(4), axis=0), 2, axis=0
    )  # shape [2, 4, 4]
    self.conditioning = dict()
    self.dummy_inference_fn = dummy_inference_fn

  def test_confident_logits_stop_after_first_update(self):
    """When logits are always very confident the sampler stops early.

    The entropy of a near-one-hot distribution over V=4 is ~0, which is
    below the default threshold of 0.05, so the sampler should stop after
    the very first update step.
    """

    # logits_fn: always return very confident logits → entropy ≈ 0.
    def confident_logits(xt):
      b, l, _ = xt.shape
      logits = jnp.full((b, l, 4), -100.0)
      return logits.at[:, :, 0].set(100.0)

    stepper = DummyStepWithLogits(logits_fn=confident_logits)
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=stepper,
        num_steps=10,
        early_stopping_fn=(
            diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
                entropy_threshold=0.05,
            )
        ),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # Should stop after 1 update + finalize = 2 total steps.
    self.assertEqual(int(last_step.step_info.step), 2)

    # Verify the output: init=eye, 1 shift_right, then finalize=invert.
    shifted = shift_right(jnp.eye(4))
    expected_single = invert(shifted)
    expected = jnp.repeat(
        jnp.expand_dims(expected_single, axis=0), repeats=2, axis=0
    )
    chex.assert_trees_all_equal(last_step.xt, expected)

  def test_uniform_logits_run_all_steps(self):
    """When logits are uniform the entropy is high and no early stop occurs."""

    # logits_fn: uniform logits → entropy = ln(V) ≈ 1.386 >> 0.05.
    def uniform_logits(xt):
      b, l, _ = xt.shape
      return jnp.zeros((b, l, 4))

    stepper = DummyStepWithLogits(logits_fn=uniform_logits)
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=stepper,
        num_steps=5,
        early_stopping_fn=(
            diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
                entropy_threshold=0.05,
            )
        ),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # All 4 updates + finalize → 5 total steps.
    self.assertEqual(int(last_step.step_info.step), 5)

  def test_per_element_entropy_stop(self):
    """Batch element 0 has confident logits (stops early), element 1 is uniform.

    Element 0 should freeze after the first update while element 1 runs
    all update steps.
    """

    def mixed_logits(xt):
      b, l, _ = xt.shape
      # Element 0: confident (near one-hot) → low entropy → stops.
      confident = jnp.full((1, l, 4), -100.0).at[:, :, 0].set(100.0)
      # Element 1: uniform → high entropy → continues.
      uniform = jnp.zeros((1, l, 4))
      return jnp.concatenate([confident, uniform], axis=0)

    stepper = DummyStepWithLogits(logits_fn=mixed_logits)
    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=stepper,
        num_steps=5,
        early_stopping_fn=(
            diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
                entropy_threshold=0.05,
            )
        ),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # Element 0: 1 shift_right (frozen after step 0), then finalize=invert.
    elem0_expected = invert(shift_right(jnp.eye(4)))
    # Element 1: 4 shift_rights (runs all steps), then finalize=invert.
    # shift_right^4 on a 4x4 matrix wraps back to identity.
    elem1_expected = invert(jnp.eye(4))

    expected = jnp.stack([elem0_expected, elem1_expected], axis=0)
    chex.assert_trees_all_equal(last_step.xt, expected)

    # All 4 updates executed (element 1 kept going) + finalize = 5.
    self.assertEqual(int(last_step.step_info.step), 5)

  def test_custom_logits_key(self):
    """Entropy early stopping reads from a custom aux key."""

    def confident_logits(xt):
      b, l, _ = xt.shape
      logits = jnp.full((b, l, 4), -100.0)
      return logits.at[:, :, 0].set(100.0)

    # Stepper puts logits under 'logits', but we'll use a custom key stepper.
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class StepWithCustomKey(SamplerStep):

      def initialize(self, initial_noise, initial_step_info):
        return base.DiffusionStep(
            xt=initial_noise,
            step_info=initial_step_info,
            aux={'my_logits': confident_logits(initial_noise)},
        )

      def update(self, prediction, current_step, next_step_info):
        new_xt = shift_right(prediction['x0'])
        return base.DiffusionStep(
            xt=new_xt,
            step_info=next_step_info,
            aux={'my_logits': confident_logits(new_xt)},
        )

      def finalize(self, prediction, current_step, next_step_info):
        new_xt = invert(prediction['x0'])
        return base.DiffusionStep(
            xt=new_xt,
            step_info=next_step_info,
            aux={'my_logits': confident_logits(new_xt)},
        )

    sample_fn = sampling.DiffusionSamplerWithEarlyStopping(
        time_schedule=self.time_schedule,
        stepper=StepWithCustomKey(),
        num_steps=10,
        early_stopping_fn=(
            diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
                entropy_threshold=0.05,
                logits_key='my_logits',
            )
        ),
    )
    last_step, _ = sample_fn(
        inference_fn=self.dummy_inference_fn,  # pyrefly: ignore[bad-argument-type]
        initial_noise=self.initial_noise,
        conditioning=self.conditioning,
        rng=jax.random.PRNGKey(0),
    )

    # Should stop after 1 update + finalize = 2.
    self.assertEqual(int(last_step.step_info.step), 2)


################################################################################
# MARK: Helper function tests
################################################################################


class FreezeDoneElementsTest(parameterized.TestCase):
  """Tests for _freeze_done_elements."""

  def _make_step_info(self):
    return base.StepInfo(
        step=jnp.int32(0),
        time=jnp.float32(0.5),
        rng=jax.random.PRNGKey(0),
    )

  def test_no_done_elements(self):
    """When no elements are done, new_step passes through unchanged."""
    new_xt = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    old_xt = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    new_step = base.DiffusionStep(
        xt=new_xt, step_info=self._make_step_info(), aux={}
    )
    old_step = base.DiffusionStep(
        xt=old_xt, step_info=self._make_step_info(), aux={}
    )
    done = jnp.array([False, False])
    result = sampling._freeze_done_elements(new_step, old_step, done)
    chex.assert_trees_all_equal(result.xt, new_xt)

  def test_all_done_elements(self):
    """When all elements are done, old_step values are kept."""
    new_xt = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    old_xt = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    new_step = base.DiffusionStep(
        xt=new_xt, step_info=self._make_step_info(), aux={}
    )
    old_step = base.DiffusionStep(
        xt=old_xt, step_info=self._make_step_info(), aux={}
    )
    done = jnp.array([True, True])
    result = sampling._freeze_done_elements(new_step, old_step, done)
    chex.assert_trees_all_equal(result.xt, old_xt)

  def test_partial_done_elements(self):
    """Only done elements are frozen; active elements use new values."""
    new_xt = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    old_xt = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    new_step = base.DiffusionStep(
        xt=new_xt, step_info=self._make_step_info(), aux={}
    )
    old_step = base.DiffusionStep(
        xt=old_xt, step_info=self._make_step_info(), aux={}
    )
    done = jnp.array([True, False])
    result = sampling._freeze_done_elements(new_step, old_step, done)
    expected_xt = jnp.array([[5.0, 6.0], [3.0, 4.0]])
    chex.assert_trees_all_equal(result.xt, expected_xt)

  def test_scalar_leaves_pass_through(self):
    """Scalar/non-batch leaves should pass through from new_step."""
    new_step = base.DiffusionStep(
        xt=jnp.array([[1.0, 2.0]]),
        step_info=base.StepInfo(
            step=jnp.int32(10),
            time=jnp.float32(0.5),
            rng=jax.random.PRNGKey(1),
        ),
        aux={},
    )
    old_step = base.DiffusionStep(
        xt=jnp.array([[5.0, 6.0]]),
        step_info=base.StepInfo(
            step=jnp.int32(5),
            time=jnp.float32(0.9),
            rng=jax.random.PRNGKey(0),
        ),
        aux={},
    )
    done = jnp.array([True])
    result = sampling._freeze_done_elements(new_step, old_step, done)
    # xt is frozen (batch dim matches done).
    chex.assert_trees_all_equal(result.xt, old_step.xt)
    # Scalar step_info.step passes through from new_step.
    self.assertEqual(int(result.step_info.step), 10)


class IndexPytreeTest(absltest.TestCase):
  """Tests for _index_pytree."""

  def test_indexes_dict(self):
    pytree = dict(a=jnp.array([10, 20, 30]), b=jnp.array([40, 50, 60]))
    result = sampling._index_pytree(pytree, 1)
    chex.assert_trees_all_equal(result, dict(a=20, b=50))

  def test_indexes_list(self):
    pytree = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]
    result = sampling._index_pytree(pytree, 2)
    chex.assert_trees_all_equal(result, [3, 6])


if __name__ == '__main__':
  absltest.main()
