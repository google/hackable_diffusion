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

"""Tests for sampling."""

import dataclasses

import chex
from hackable_diffusion.lib.sampling import base
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
# MARK: Helper functions
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


################################################################################
# MARK: Tests
################################################################################


class DiffusionSamplingTest(parameterized.TestCase):

  # MARK: Test for helper functions

  def setUp(self):
    super().setUp()
    self.time_schedule = time_scheduling.UniformTimeSchedule(safety_epsilon=0.0)
    self.stepper = DummyStep()
    self.initial_noise = jnp.repeat(
        jnp.expand_dims(jnp.eye(4), axis=0), 2, axis=0
    )
    self.conditioning = dict()
    self.dummy_inference_fn = dummy_inference_fn

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

  def test_concat_pytree_invalid_tree(self):
    with self.assertRaisesRegex(ValueError, 'Dict key mismatch'):
      sampling._concat_pytree(dict(a=1), dict(a=2, b=3), dict(a=4))

  # MARK: Test for diffusion_sampling

  def test_sample_one(self):
    """Test the sampling function on a toy example."""

    sample_fn = sampling.DiffusionSampler(
        time_schedule=self.time_schedule,
        stepper=self.stepper,
        num_steps=5,
    )
    last_step, all_steps = sample_fn(
        inference_fn=self.dummy_inference_fn,
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
          inference_fn=self.dummy_inference_fn,
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
        inference_fn=self.dummy_inference_fn,
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


if __name__ == '__main__':
  absltest.main()
