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

"""Tests for Gaussian corruption processes."""

import itertools

from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Constants
################################################################################


_PRED_TYPES = ('x0', 'epsilon', 'score', 'velocity', 'v')

################################################################################
# MARK: PredictionConverterTest
################################################################################


class PredictionConverterTest(parameterized.TestCase):

  @parameterized.named_parameters(*(
      {
          'testcase_name': f'{start_type}_to_{interm_type}_to_{start_type}',
          'start_type': start_type,
          'interm_type': interm_type,
      }
      for (start_type, interm_type) in itertools.product(
          _PRED_TYPES, _PRED_TYPES
      )
  ))
  def test_cycle_consistency(self, start_type, interm_type):
    # Tolerances for numerical comparison
    atol = 1e-4
    rtol = 1e-4
    shape = (11, 7, 5, 3)  # Example dimensions
    t = jnp.linspace(0.05, 0.95, shape[0]).reshape((-1, 1, 1, 1))

    schedule = schedules.CosineSchedule()

    key_xt, key_initial_val = jax.random.split(jax.random.PRNGKey(42))
    xt = jax.random.uniform(key_xt, shape, minval=-1.0, maxval=1.0)
    initial_val = jax.random.uniform(
        key_initial_val, shape, minval=-1.0, maxval=1.0
    )

    sigma = schedule.sigma(t)
    alpha = schedule.alpha(t)
    alpha_der = utils.egrad(schedule.alpha)(t)
    sigma_der = utils.egrad(schedule.sigma)(t)

    kwargs = {
        'xt': xt,
        'alpha': alpha,
        'sigma': sigma,
        'alpha_der': alpha_der,
        'sigma_der': sigma_der,
    }
    interm_val = gaussian.CONVERTERS[start_type][interm_type](
        initial_val, **kwargs
    )  # pytype: disable=wrong-keyword-args
    recomputed_start_val = gaussian.CONVERTERS[interm_type][start_type](
        interm_val, **kwargs
    )  # pytype: disable=wrong-keyword-args

    self.assertTrue(
        jnp.allclose(initial_val, recomputed_start_val, atol=atol, rtol=rtol),
        f'Cycle consistency failed: {start_type} -> {interm_type} ->'
        f' {start_type}. Max absolute difference:'
        f' {jnp.max(jnp.abs(initial_val - recomputed_start_val))}',
    )


if __name__ == '__main__':
  absltest.main()
