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

from hackable_diffusion.lib import time_sampling
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized


class TimeSamplersTest(parameterized.TestCase):

  def test_uniform_time_sampler(self):
    data_shape = jnp.zeros((2, 3, 4))
    key = jax.random.PRNGKey(0)

    # Test with default batch_axes
    sampler = time_sampling.UniformTimeSampler()
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (2, 1, 1))
    self.assertTrue(jnp.all(time >= 0.0))
    self.assertTrue(jnp.all(time <= 1.0))

    # Test with different batch_axes
    sampler = time_sampling.UniformTimeSampler(axes=(1,))
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (1, 3, 1))
    self.assertTrue(jnp.all(time >= 0.0))
    self.assertTrue(jnp.all(time <= 1.0))

    # Test with different time_range
    sampler = time_sampling.UniformTimeSampler(time_range=(0.2, 0.8))
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (2, 1, 1))
    self.assertTrue(jnp.all(time >= 0.2))
    self.assertTrue(jnp.all(time <= 0.8))

  def test_from_safety_epsilon(self):
    sampler = time_sampling.UniformTimeSampler.from_safety_epsilon(
        safety_epsilon=0.1
    )
    self.assertEqual(sampler.time_range, (0.1, 0.9))

  @parameterized.named_parameters(
      dict(
          testcase_name="uniform",
          sampler_cls=time_sampling.UniformTimeSampler,
          sampler_kwargs={},
      ),
      dict(
          testcase_name="uniform_stratified",
          sampler_cls=time_sampling.UniformStratifiedTimeSampler,
          sampler_kwargs={},
      ),
  )
  def test_time_sampler(self, sampler_cls, sampler_kwargs):
    key = jax.random.PRNGKey(0)
    data_shape = jnp.zeros((2, 3, 5))

    sampler = sampler_cls(**sampler_kwargs)
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (2, 1, 1))
    self.assertTrue(jnp.all(time >= sampler.time_range[0]))
    self.assertTrue(jnp.all(time <= sampler.time_range[1]))

    # Test with different axes
    sampler = sampler_cls(**sampler_kwargs, axes=(1, 2))
    time = sampler(key, data_shape)
    self.assertEqual(time.shape, (1, 3, 5))

  def test_nested_time_sampler(self):
    key = jax.random.PRNGKey(0)
    data_spec = {
        "image": jnp.zeros((2, 3, 4)),
        "label": jnp.zeros((2,)),
    }

    sampler = time_sampling.NestedTimeSampler(
        samplers={
            "image": time_sampling.UniformTimeSampler(axes=(0, 1)),
            "label": time_sampling.UniformTimeSampler(),
        }
    )
    time = sampler(key, data_spec)

    self.assertIsInstance(time, dict)
    self.assertEqual(time["image"].shape, (2, 3, 1))
    self.assertEqual(time["label"].shape, (2,))


if __name__ == "__main__":
  absltest.main()
