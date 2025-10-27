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

"""Tests for time scheduling."""

import chex
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

from absl.testing import absltest

################################################################################
# MARK: Tests
################################################################################


class TimeScheduleTest(absltest.TestCase):

  # MARK: UniformTimeSchedule tests

  def test_uniform_all_step_infos(self):
    time_schedule = time_scheduling.UniformTimeSchedule(safety_epsilon=0.1)
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[0.9], [0.9]],
        [[0.7], [0.7]],
        [[0.5], [0.5]],
        [[0.3], [0.3]],
        [[0.1], [0.1]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  def test_uniform_all_step_infos_without_safety_epsilon(self):
    time_schedule = time_scheduling.UniformTimeSchedule(safety_epsilon=0.0)
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[1.0], [1.0]],
        [[0.75], [0.75]],
        [[0.5], [0.5]],
        [[0.25], [0.25]],
        [[0.0], [0.0]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  def test_fail_epsilon_out_of_range(self):
    with self.assertRaisesRegex(ValueError, "must be between 0.0 and 1.0"):
      time_scheduling.UniformTimeSchedule(safety_epsilon=-0.1)

    with self.assertRaisesRegex(ValueError, "must be between 0.0 and 1.0"):
      time_scheduling.UniformTimeSchedule(safety_epsilon=1.1)

  # MARK: EDMTimeSchedule tests

  def test_all_step_infos(self):
    time_schedule = time_scheduling.EDMTimeSchedule(safety_epsilon=0.0, rho=2.0)
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[1.0], [1.0]],
        [[0.5625], [0.5625]],
        [[0.25], [0.25]],
        [[0.0625], [0.0625]],
        [[0.0], [0.0]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  def test_edm_all_step_infos_with_rho_one_is_uniform(self):
    uniform_time_schedule = time_scheduling.UniformTimeSchedule(
        safety_epsilon=0.1
    )
    edm_time_schedule = time_scheduling.EDMTimeSchedule(
        safety_epsilon=0.1, rho=1.0
    )
    data_spec = jnp.zeros((2, 3))
    num_steps = 5
    uniform_steps = uniform_time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=num_steps,
        data_spec=data_spec,
    ).time
    edm_steps = edm_time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=num_steps,
        data_spec=data_spec,
    ).time
    chex.assert_trees_all_close(uniform_steps, edm_steps)

  def test_nested_time_schedule(self):
    # Create a nested time schedule
    time_schedule_continuous = time_scheduling.UniformTimeSchedule()
    time_schedule_discrete = time_scheduling.UniformTimeSchedule()
    time_schedules = {
        "data_continuous": time_schedule_continuous,
        "modality": {"data_discrete": time_schedule_discrete},
    }
    time_schedule = time_scheduling.NestedTimeSchedule(
        time_schedules=time_schedules
    )

    data_spec = {
        "data_continuous": jnp.zeros((2, 3)),
        "modality": {"data_discrete": jnp.zeros((2, 4, 5))},
    }
    num_steps = 5
    time_info = time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=num_steps,
        data_spec=data_spec,
    )
    self.assertIsInstance(time_info, dict)
    self.assertEqual(time_info["data_continuous"].time.shape, (5, 2, 1))
    self.assertEqual(
        time_info["modality"]["data_discrete"].time.shape, (5, 2, 1, 1)
    )
    chex.assert_trees_all_equal_structs(time_schedules, data_spec)


if __name__ == "__main__":
  absltest.main()
