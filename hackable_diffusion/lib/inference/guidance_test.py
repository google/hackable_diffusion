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

"""Tests for guidance functions."""

from hackable_diffusion.lib import corruption
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.inference import guidance
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Tests
################################################################################


class GuidanceTest(parameterized.TestCase):
  """Tests for guidance functions."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.data_shape = (4, 4, 3)
    self.xt = jnp.ones((self.batch_size, *self.data_shape))
    self.conditioning = {}  # Not used by these guidance functions
    self.cond_outputs = {'pred': jnp.ones_like(self.xt) * 2.0}
    self.uncond_outputs = {'pred': jnp.ones_like(self.xt) * 1.0}

    self.guidance_val = 3.0
    self.lower_logsnr = -9.0
    self.upper_logsnr = 9.0
    self.lower_time = 0.25
    self.upper_time = 0.75

    # nested versions of the above
    self.nested_xt = {
        'data_continuous': jnp.ones((self.batch_size, *self.data_shape)),
        'modality': {
            'data_discrete': jnp.ones((self.batch_size, *self.data_shape)),
        },
    }
    self.nested_cond_outputs = {
        'data_continuous': {'pred': jnp.ones_like(self.xt) * 2.0},
        'modality': {
            'data_discrete': {'pred': jnp.ones_like(self.xt) * 2.0},
        },
    }
    self.nested_uncond_outputs = {
        'data_continuous': {'pred': jnp.ones_like(self.xt) * 1.0},
        'modality': {
            'data_discrete': {'pred': jnp.ones_like(self.xt) * 1.0},
        },
    }

  def test_scalar_guidance_fn(self):
    """Tests the ScalarGuidanceFn."""
    guidance_fn = guidance.ScalarGuidanceFn(guidance=self.guidance_val)
    time = jnp.array([0.5, 0.5])  # Not used, but required by protocol

    result = guidance_fn(
        self.xt, self.conditioning, time, self.cond_outputs, self.uncond_outputs
    )

    # Expected: cond * (1 + g) - uncond * g = 2.0 * 4.0 - 1.0 * 3.0 = 5.0
    expected_output = jnp.ones_like(self.xt) * 5.0
    self.assertTrue(jnp.allclose(result['pred'], expected_output))

  def test_limited_interval_guidance_fn_fails_on_invalid_interval(self):
    """Tests the LimitedIntervalGuidanceFn with a batch of times."""
    # First time is outside the interval, second is inside.
    time = jnp.array([0.1, 0.5])
    time = jax_helpers.bcast_right(time, self.xt.ndim)
    with self.assertRaisesRegex(
        ValueError,
        'Lower bound must be strictly smaller than the upper bound.',
    ):
      guidance_fn = guidance.LimitedIntervalGuidanceFn(
          guidance=self.guidance_val,
          lower=self.upper_time,
          upper=self.lower_time,
      )
      guidance_fn(
          self.xt,
          self.conditioning,
          time,
          self.cond_outputs,
          self.uncond_outputs,
      )

  def test_limited_interval_guidance_fn(self):
    """Tests the LimitedIntervalGuidanceFn with a batch of times."""
    guidance_fn = guidance.LimitedIntervalGuidanceFn(
        guidance=self.guidance_val,
        lower=self.lower_time,
        upper=self.upper_time,
    )
    # First time is outside the interval, second is inside.
    time = jnp.array([0.1, 0.5])
    time = jax_helpers.bcast_right(time, self.xt.ndim)

    result = guidance_fn(
        self.xt, self.conditioning, time, self.cond_outputs, self.uncond_outputs
    )

    # For time=0.1 (outside), guidance is 0. Output should be cond_output.
    # Expected: 2.0 * (1 + 0) - 1.0 * 0 = 2.0
    # For time=0.5 (inside), guidance is 3.0.
    # Expected: 2.0 * (1 + 3) - 1.0 * 3 = 5.0
    expected_output = jnp.stack([
        jnp.ones(self.data_shape) * 2.0,
        jnp.ones(self.data_shape) * 5.0,
    ])
    self.assertTrue(jnp.allclose(result['pred'], expected_output))

  def test_limited_logsnr_interval_guidance_fn(self):
    """Tests the LimitedIntervalGuidanceFn with a batch of times."""
    guidance_fn = guidance.LogSnrLimitedIntervalGuidanceFn(
        noise_schedule=corruption.RFSchedule(),
        guidance=self.guidance_val,
        lower=self.lower_logsnr,
        upper=self.upper_logsnr,
    )
    # First time is outside the interval, second is inside.
    time = jnp.array([0.1, 0.99])
    time = jax_helpers.bcast_right(time, self.xt.ndim)

    result = guidance_fn(
        self.xt, self.conditioning, time, self.cond_outputs, self.uncond_outputs
    )

    # For time=0.1, logsnr is ~4.4, so it's inside the interval, guidance is 3.0
    # Expected: 2.0 * (1 + 3) - 1.0 * 3 = 5.0
    # For time=0.99, logsnr is ~-9.19, so it's outside the interval,
    # guidance is 0.
    # Expected: 2.0 * (1 + 0) - 1.0 * 0 = 2.0
    expected_output = jnp.stack([
        jnp.ones(self.data_shape) * 5.0,
        jnp.ones(self.data_shape) * 2.0,
    ])
    self.assertTrue(jnp.allclose(result['pred'], expected_output))


if __name__ == '__main__':
  absltest.main()
