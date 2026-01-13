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

"""Tests for Gaussian loss functions."""

import itertools

from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import gaussian as gaussian_corrupt
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.loss import gaussian
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Constants
################################################################################

_PRED_TYPES = ('x0', 'epsilon', 'score', 'velocity', 'v')


################################################################################
# MARK: Tests
################################################################################


class GaussianLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bsz = 4
    self.shape = (self.bsz, 8, 8, 3)
    self.key = jax.random.PRNGKey(42)

    key_pred, key_target = jax.random.split(self.key)

    self.preds = {
        'x0': jax.random.uniform(key_pred, self.shape, minval=-1.0, maxval=1.0)
    }
    self.targets = {
        'x0': jax.random.uniform(
            key_target, self.shape, minval=-1.0, maxval=1.0
        )
    }
    self.time = jnp.ones((self.bsz, 1, 1, 1)) * 0.5
    self.schedule = schedules.CosineSchedule()

  @parameterized.named_parameters(
      ('no_weight', gaussian.NoWeightGaussianLoss, {}, 1e-6),
      ('sid2_loss_no_bias', gaussian.SiD2Loss, {'bias': 0.0}, 1e-6),
      ('sid2_loss_with_bias', gaussian.SiD2Loss, {'bias': 1.0}, 1e-6),
  )
  def test_loss_computation(self, loss_class, loss_kwargs, atol):
    """Tests various Gaussian loss computations."""
    final_kwargs = loss_kwargs.copy()
    if loss_class is gaussian.SiD2Loss:
      final_kwargs['schedule'] = self.schedule

    loss_fn = loss_class(**final_kwargs)
    loss = loss_fn(self.preds, self.targets, self.time)

    self.assertEqual(loss.shape, (self.bsz,))
    self.assertFalse(jnp.isnan(loss).any())
    self.assertTrue(jnp.all(loss >= 0.0))

    # Manual computation of expected loss
    pred = self.preds['x0']
    target = self.targets['x0']
    l2 = jnp.square(pred - target)

    weight = jnp.ones(self.bsz)
    if loss_class is gaussian.NoWeightGaussianLoss:
      pass  # Weight is 1
    elif loss_class is gaussian.SiD2Loss:
      bias = loss_kwargs.get('bias', 0.0)
      logsnr = self.schedule.logsnr(self.time)
      logsnr_der = utils.egrad(self.schedule.logsnr)(self.time)
      weight_from_fn = jax.nn.sigmoid(logsnr - bias) * jnp.exp(bias)
      # SiD2Loss has convert_to_logsnr_schedule=True and loss_type='x0'
      # Since we provide 'x0' preds, conversion term is 1.
      weight = -logsnr_der * weight_from_fn

    reshaped_weight = weight.reshape((self.bsz,) + (1,) * (pred.ndim - 1))
    weighted_l2 = reshaped_weight * l2
    expected_loss = jnp.mean(weighted_l2, axis=tuple(range(1, pred.ndim)))

    self.assertTrue(jnp.allclose(loss, expected_loss, atol=atol))


class PredictionConverterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.time = jnp.linspace(0.05, 0.95, 11).reshape((-1, 1, 1, 1))
    self.shape = (11, 7, 5, 3)
    key_x0, key_pred = jax.random.split(jax.random.PRNGKey(42), 2)
    target = jax.random.uniform(key_x0, self.shape, minval=-1.0, maxval=1.0)
    pred = jax.random.uniform(key_pred, self.shape, minval=-1.0, maxval=1.0)
    self.preds = {'x0': pred}
    self.targets = {'x0': target}

  @parameterized.named_parameters(*(
      {
          'testcase_name': f'{start_type}_to_{target_type}',
          'start_type': start_type,
          'target_type': target_type,
      }
      for (start_type, target_type) in itertools.product(
          _PRED_TYPES, _PRED_TYPES
      )
  ))
  def test_convert_and_scale_are_equivalent(self, start_type, target_type):
    # The idea of this test is to check that the scaled loss is numerically
    # equivalent to the converted loss. So:
    #
    # (pred - target)**2 * scale_term
    #             ==
    # (convert(pred) - convert(target))**2

    # Tolerances for numerical comparison
    atol = 1e-4
    rtol = 1e-4
    shape = (11, 7, 5, 3)  # Example dimensions
    time = jnp.linspace(0.05, 0.95, shape[0]).reshape((-1, 1, 1, 1))

    schedule = schedules.CosineSchedule()

    # Generate random data for target, xt and pred
    key_target, key_xt, key_pred = jax.random.split(jax.random.PRNGKey(42), 3)
    target = jax.random.uniform(key_target, shape, minval=-1.0, maxval=1.0)
    xt = jax.random.uniform(key_xt, shape, minval=-1.0, maxval=1.0)
    pred = jax.random.uniform(key_pred, shape, minval=-1.0, maxval=1.0)

    sigma = schedule.sigma(time)
    alpha = schedule.alpha(time)
    alpha_der = utils.egrad(schedule.alpha)(time)
    sigma_der = utils.egrad(schedule.sigma)(time)

    kwargs = {
        'xt': xt,
        'alpha': alpha,
        'sigma': sigma,
        'alpha_der': alpha_der,
        'sigma_der': sigma_der,
    }

    # Get the converter
    converter = gaussian_corrupt.CONVERTERS[start_type][target_type]

    converted_pred = converter(pred, **kwargs)
    converted_target = converter(target, **kwargs)
    scaled_loss = gaussian.compute_continuous_diffusion_loss(
        preds={start_type: pred},
        targets={start_type: target},
        time=time,
        schedule=schedule,
        loss_type=target_type,
        prediction_type=start_type,
        # start_type != target_type trigger scaling
    )
    converted_loss = gaussian.compute_continuous_diffusion_loss(
        preds={target_type: converted_pred},
        targets={target_type: converted_target},
        time=time,
        schedule=schedule,
        loss_type=target_type,
        prediction_type=target_type,
    )

    self.assertTrue(
        jnp.allclose(scaled_loss, converted_loss, atol=atol, rtol=rtol),
        f'Loss conversion check failed: {start_type} -> {target_type}\n'
        'Absolute difference:'
        f' {jnp.max(jnp.abs(scaled_loss - converted_loss))}',
    )

  @parameterized.named_parameters(
      ('with_convert_to_logsnr', True, None),
      ('with_weight_fn', False, lambda schedule, preds, targets, time: 1.0),
  )
  def test_raises_error_if_schedule_is_none_but_required(
      self, convert_to_logsnr_schedule, weight_fn
  ):
    """Tests that an error is raised if schedule is None but required."""
    with self.assertRaisesRegex(
        ValueError,
        'Schedule must be provided if convert_to_logsnr_schedule or weight_fn'
        ' is not None.',
    ):
      gaussian.compute_continuous_diffusion_loss(
          preds=self.preds,
          targets=self.targets,
          time=self.time,
          schedule=None,
          convert_to_logsnr_schedule=convert_to_logsnr_schedule,
          weight_fn=weight_fn,
      )


if __name__ == '__main__':
  absltest.main()
