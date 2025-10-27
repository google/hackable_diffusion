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

"""Tests for guidance functions."""

from hackable_diffusion.lib import utils
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
    guidance_val = 3.0
    guidance_fn = guidance.ScalarGuidanceFn(guidance=guidance_val)
    time = jnp.array([0.5, 0.5])  # Not used, but required by protocol

    result = guidance_fn(
        self.xt, self.conditioning, time, self.cond_outputs, self.uncond_outputs
    )

    # Expected: cond * (1 + g) - uncond * g = 2.0 * 4.0 - 1.0 * 3.0 = 5.0
    expected_output = jnp.ones_like(self.xt) * 5.0
    self.assertTrue(jnp.allclose(result['pred'], expected_output))

  def test_limited_interval_guidance_fn_fails_on_invalid_interval(self):
    """Tests the LimitedIntervalGuidanceFn with a batch of times."""
    guidance_val = 3.0
    lower = 0.75
    upper = 0.25
    # First time is outside the interval, second is inside.
    time = jnp.array([0.1, 0.5])
    time = utils.bcast_right(time, self.xt.ndim)
    with self.assertRaisesRegex(
        ValueError,
        'Lower bound must be strictly smaller than the upper bound.',
    ):
      guidance_fn = guidance.LimitedIntervalGuidanceFn(
          guidance=guidance_val,
          lower=lower,
          upper=upper,
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
    guidance_val = 3.0
    lower = 0.25
    upper = 0.75
    guidance_fn = guidance.LimitedIntervalGuidanceFn(
        guidance=guidance_val,
        lower=lower,
        upper=upper,
    )
    # First time is outside the interval, second is inside.
    time = jnp.array([0.1, 0.5])
    time = utils.bcast_right(time, self.xt.ndim)

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

  def test_nested_guidance_fn(self):
    """Tests the NestedGuidanceFn."""
    guidance_val = 3.0
    guidance_fn = guidance.NestedGuidanceFn(
        guidance_fns={
            'data_continuous': guidance.ScalarGuidanceFn(guidance=guidance_val),
            'modality': {
                'data_discrete': guidance.ScalarGuidanceFn(
                    guidance=guidance_val
                ),
            },
        }
    )
    time = {
        'data_continuous': jnp.array([0.5, 0.5]),
        'modality': {
            'data_discrete': jnp.array([0.5, 0.5]),
        },
    }  # Not used, but required by protocol
    result = guidance_fn(
        self.nested_xt,
        self.conditioning,
        time,
        self.nested_cond_outputs,
        self.nested_uncond_outputs,
    )
    self.assertIsInstance(result, dict)
    self.assertEqual(
        result['data_continuous']['pred'].shape,
        (self.batch_size, *self.data_shape),
    )
    self.assertEqual(
        result['modality']['data_discrete']['pred'].shape,
        (self.batch_size, *self.data_shape),
    )


if __name__ == '__main__':
  absltest.main()
