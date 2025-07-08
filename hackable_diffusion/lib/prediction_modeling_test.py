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

from hackable_diffusion.lib import noise_scheduling
from hackable_diffusion.lib import prediction_modeling

import jax
import jax.numpy as jnp

from absl.testing import absltest


class PredictionModelTest(absltest.TestCase):

  def test_x0_to_score(self):
    noise_schedule = noise_scheduling.RFSchedule()
    x0, t, xt = jnp.ones((2, 2)), 0.5, jnp.zeros((2, 2))

    score = prediction_modeling.x0_to_score(noise_schedule, x0, t, xt)

    expected_score = jnp.full_like(x0, 2.0)
    self.assertTrue(jnp.allclose(score, expected_score))


class IdentityPredictionModelTest(absltest.TestCase):

  def test_predict(self):
    predictor = prediction_modeling.IdentityPredictionModel()

    chex.assert_trees_all_equal(
        predictor.predict(
            xt=jnp.eye(2),
            conditioning={},
            t=0.5,
            rng=jax.random.PRNGKey(0),
        ),
        dict(x0=jnp.eye(2)),
    )

if __name__ == "__main__":
  absltest.main()
