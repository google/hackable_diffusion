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

import chex

from hackable_diffusion.lib import time_scheduling

import jax.numpy as jnp

from absl.testing import absltest


class UniformTimeScheduleTest(absltest.TestCase):

  def test_eval_schedule(self):
    time_schedule = time_scheduling.UniformTimeSchedule(safety_epsilon=0.1)
    chex.assert_trees_all_close(
        time_schedule.eval_schedule(5),
        jnp.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    )

  def test_eval_schedule_without_safety_epsilon(self):
    time_schedule = time_scheduling.UniformTimeSchedule(safety_epsilon=0.0)
    chex.assert_trees_all_close(
        time_schedule.eval_schedule(5),
        jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]),
    )

  def test_fail_epsilon_out_of_range(self):
    with self.assertRaisesRegex(ValueError, "must be between 0.0 and 1.0"):
      time_scheduling.UniformTimeSchedule(safety_epsilon=-.1)

    with self.assertRaisesRegex(ValueError, "must be between 0.0 and 1.0"):
      time_scheduling.UniformTimeSchedule(safety_epsilon=1.1)

if __name__ == "__main__":
  absltest.main()
