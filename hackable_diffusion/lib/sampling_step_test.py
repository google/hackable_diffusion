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
from hackable_diffusion.lib import sampling_step

import jax
import jax.numpy as jnp

from absl.testing import absltest


class SdeStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.sde_step = sampling_step.SdeStep(
        noise_schedule=noise_scheduling.RFSchedule(), churn=0.1
    )

  def test_initialize(self):

    predictor = prediction_modeling.IdentityPredictionModel()
    initial_step = self.sde_step.initialize(
        predictor=predictor,
        initial_noise=jnp.eye(4),
        conditioning=dict(),
        initial_step_info=dict(step=0, t=0.0, rng=jax.random.PRNGKey(0)),
    )

    chex.assert_trees_all_equal(
        initial_step,
        dict(
            xt=jnp.eye(4),
            prediction=dict(x0=jnp.eye(4)),
            conditioning=dict(),
            step_info=dict(step=0, t=0., rng=jax.random.PRNGKey(0)),
            aux=dict(f=jnp.nan, g=jnp.nan, mean=jnp.full((4, 4), jnp.nan),
                     volatility=jnp.nan),
        )
    )

  def test_update(self):

    predictor = prediction_modeling.IdentityPredictionModel()
    initial_step = self.sde_step.initialize(
        predictor=predictor,
        initial_noise=jnp.eye(4),
        conditioning=dict(),
        initial_step_info=dict(step=0, t=0.1, rng=jax.random.PRNGKey(0)),
    )

    next_step = self.sde_step.update(
        predictor=predictor,
        current_step=initial_step,
        next_step_info=dict(step=1, t=0.2, rng=jax.random.PRNGKey(1)),
    )

    chex.assert_trees_all_close(
        next_step,
        {
            'xt': jnp.array(
                [
                    [0.9965866, 0.00126274, -0.00202708, -0.00231114],
                    [0.01888236, 1.0010996, 0.0319245, 0.01494699],
                    [-0.00432807, 0.00534189, 0.98833567, -0.00366052],
                    [0.01320149, 0.01171877, 0.01325134, 1.0070777],
                ],
                dtype=jnp.float32,
            ),
            'prediction': {
                'x0': jnp.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=jnp.float32,
                )
            },
            'conditioning': {},
            'step_info': {
                'step': 1,
                't': 0.2,
                'rng': jax.random.PRNGKey(1),
            },
            'aux': {
                'f': jnp.array(-1.1111112, dtype=jnp.float32),
                'g': jnp.array(0.47140452, dtype=jnp.float32),
                'mean': jnp.array(
                    [
                        [0.99888885, 0.0, 0.0, 0.0],
                        [0.0, 0.99888885, 0.0, 0.0],
                        [0.0, 0.0, 0.99888885, 0.0],
                        [0.0, 0.0, 0.0, 0.99888885],
                    ],
                    dtype=jnp.float32,
                ),
                'volatility': jnp.array(0.01490712, dtype=jnp.float32),
            },
        },
        atol=1e-6,
    )

  def test_finalize(self):
    # TODO(ccrepy): Implement and once the lib is available.
    pass


if __name__ == "__main__":
  absltest.main()
