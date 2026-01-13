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

"""Test for time sampling."""

import chex
from hackable_diffusion.lib import time_sampling
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Tests
################################################################################


class TimeSamplersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="uniform",
          sampler_cls=time_sampling.UniformTimeSampler,
      ),
      dict(
          testcase_name="logit_normal",
          sampler_cls=time_sampling.LogitNormalTimeSampler,
      ),
      dict(
          testcase_name="uniform_stratified",
          sampler_cls=time_sampling.UniformStratifiedTimeSampler,
      ),
  )
  def test_time_sampler(self, sampler_cls):
    data_shape = jnp.zeros((2, 3, 4))
    key = jax.random.PRNGKey(0)

    # Test with default batch_axes
    sampler = sampler_cls()
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (2, 1, 1))
    self.assertTrue(jnp.all(time >= 0.0))
    self.assertTrue(jnp.all(time <= 1.0))

    # Test with different batch_axes
    sampler = sampler_cls(axes=(0, 1))
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (2, 3, 1))
    self.assertTrue(jnp.all(time >= 0.0))
    self.assertTrue(jnp.all(time <= 1.0))

    # Test with different time_range
    sampler = sampler_cls(time_range=(0.2, 0.8))
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (2, 1, 1))
    self.assertTrue(jnp.all(time >= 0.2))
    self.assertTrue(jnp.all(time <= 0.8))

  @parameterized.named_parameters(
      dict(
          testcase_name="uniform",
          sampler_cls=time_sampling.UniformTimeSampler,
      ),
      dict(
          testcase_name="logit_normal",
          sampler_cls=time_sampling.LogitNormalTimeSampler,
      ),
      dict(
          testcase_name="uniform_stratified",
          sampler_cls=time_sampling.UniformStratifiedTimeSampler,
      ),
  )
  def test_from_safety_epsilon(self, sampler_cls):
    sampler = sampler_cls(safety_epsilon=0.4)
    data_shape = jnp.zeros((100, 2, 3))
    key = jax.random.PRNGKey(0)
    time = sampler(key, data_shape)
    self.assertGreaterEqual(jnp.min(time), 0.4)
    self.assertLessEqual(jnp.max(time), 0.6)

  def test_nested_time_sampler(self):
    key = jax.random.PRNGKey(0)
    data_spec = {
        "image": jnp.zeros((2, 3, 4)),
        "modality": {"label": jnp.zeros((2,))},
    }

    sampler = time_sampling.NestedTimeSampler(
        samplers={
            "image": time_sampling.UniformTimeSampler(axes=(0, 1)),
            "modality": {"label": time_sampling.UniformTimeSampler()},
        }
    )
    time = sampler(key, data_spec)

    self.assertIsInstance(time, dict)
    self.assertEqual(time["image"].shape, (2, 3, 1))
    self.assertEqual(time["modality"]["label"].shape, (2,))

  def test_joint_nested_time_sampler(self):
    """Test that the joint nested time sampler returns the same time for all modalities."""

    key = jax.random.PRNGKey(0)
    data_spec = {
        "image": jnp.zeros((2, 3, 4)),
        "modality": {"label": jnp.zeros((2,))},
    }

    sampler = time_sampling.JointNestedTimeSampler(
        samplers={
            "image": time_sampling.UniformTimeSampler(),
            "modality": {"label": time_sampling.UniformTimeSampler()},
        }
    )
    time = sampler(key, data_spec)

    self.assertIsInstance(time, dict)
    self.assertEqual(time["image"].shape, (2, 1, 1))
    self.assertEqual(time["modality"]["label"].shape, (2,))
    chex.assert_trees_all_close(
        time["image"].squeeze(),
        time["modality"]["label"].squeeze(),
    )
    chex.assert_trees_all_equal_structs(time, data_spec)


if __name__ == "__main__":
  absltest.main()
