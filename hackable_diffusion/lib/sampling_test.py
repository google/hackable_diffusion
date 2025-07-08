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

import dataclasses

import chex

from hackable_diffusion.lib import denoising
from hackable_diffusion.lib import prediction_modeling
from hackable_diffusion.lib import sampling
from hackable_diffusion.lib import time_scheduling

import jax
import jax.numpy as jnp

from absl.testing import absltest


class DiffusionSamplingTest(absltest.TestCase):

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
            )
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

  def test_sample_one(self):
    """Test the sampling function on a toy example."""

    shift_right = lambda x: jnp.roll(x, 1, axis=1)
    invert = lambda x: 1. - x

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class DummyStep(denoising.SamplerStep):

      def initialize(
          self, predictor, initial_noise, conditioning, initial_step_info
      ):
        initial_prediction = predictor.predict(
            initial_noise,
            conditioning,
            initial_step_info['t'],
            initial_step_info['rng'],
        )
        return denoising.DiffusionStep(
            xt=initial_noise,
            prediction=initial_prediction,
            conditioning=conditioning,
            step_info=initial_step_info,
            aux=dict(),
        )

      def update(self, predictor, current_step, next_step_info):
        prediction = predictor.predict(
            current_step['xt'],
            current_step['conditioning'],
            next_step_info['t'],
            next_step_info['rng'],
        )
        return denoising.DiffusionStep(
            xt=shift_right(prediction['x0']),
            prediction=prediction,
            conditioning=current_step['conditioning'],
            step_info=next_step_info,
            aux=dict(),
        )

      def finalize(self, predictor, current_step, next_step_info):
        prediction = predictor.predict(
            current_step['xt'],
            current_step['conditioning'],
            next_step_info['t'],
            next_step_info['rng'],
        )
        return denoising.DiffusionStep(
            xt=invert(prediction['x0']),
            prediction=prediction,
            conditioning=current_step['conditioning'],
            step_info=next_step_info,
            aux=dict(),
        )
    time_schedule = time_scheduling.UniformTimeSchedule(safety_epsilon=0.0)
    predictor = prediction_modeling.IdentityPredictionModel()
    stepper = DummyStep()
    initial_noise = jnp.eye(4)

    last_step, all_steps = sampling.sample_one(
        time_schedule=time_schedule,
        predictor=predictor,
        stepper=stepper,
        initial_noise=initial_noise,
        conditioning=dict(),
        num_step=5,
        rng=jax.random.PRNGKey(0),
    )

    # confirm that all steps have the correct xt
    all_xt = all_steps['xt']
    chex.assert_trees_all_equal(
        all_xt,
        jnp.array([
            [  # step 0 - init
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            [  # step 1 - shift right
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.],
            ],
            [  # step 2 - shift right
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
            ],
            [  # step 3 - shift right
                [0., 0., 0., 1.],
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
            ],
            [  # step 4 - invert
                [1., 1., 1., 0.],
                [0., 1., 1., 1.],
                [1., 0., 1., 1.],
                [1., 1., 0., 1.],
            ],
        ]),
    )

    # confirm that the last step is the same as the carry
    chex.assert_trees_all_equal(all_xt[-1], last_step['xt'])


if __name__ == "__main__":
  absltest.main()
