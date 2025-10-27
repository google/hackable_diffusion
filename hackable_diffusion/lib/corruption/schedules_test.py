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

"""Tests for schedules."""

import dataclasses
import functools
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Helper functions
################################################################################


def _identity_logsnr(schedule, x):
  return schedule.logsnr(schedule.inverse_logsnr(x))


def _identity_time(schedule, x):
  return schedule.time_change(schedule.inverse_time_change(x))


################################################################################
# MARK: Tests
################################################################################


class GaussianNoiseScheduleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    @dataclasses.dataclass(frozen=True)
    class DummyNoiseSchedule(schedules.GaussianSchedule):

      alpha_arg_history, sigma_arg_history = [], []

      def alpha(self, t):
        self.alpha_arg_history.append(t)
        return 1.0 - t

      def sigma(self, t):
        self.sigma_arg_history.append(t)
        return t

    self.fake_process = DummyNoiseSchedule()

  def test_logsnr(self):
    _ = self.fake_process.logsnr(jnp.array([1.0]))

    self.assertLen(self.fake_process.alpha_arg_history, 1)
    self.assertLen(self.fake_process.sigma_arg_history, 1)

  def test_grad_logsnr(self):
    _ = jax.grad(lambda t: jnp.sum(self.fake_process.logsnr(t)))(
        jnp.array([1.0])
    )

    self.assertLen(self.fake_process.alpha_arg_history, 1)
    self.assertLen(self.fake_process.sigma_arg_history, 1)

  def test_f(self):
    _ = self.fake_process.f(jnp.array([1.0]))

    self.assertLen(self.fake_process.alpha_arg_history, 2)
    self.assertEmpty(self.fake_process.sigma_arg_history)

  def test_g(self):
    _ = self.fake_process.g(jnp.array([1.0]))

    self.assertLen(self.fake_process.alpha_arg_history, 1)
    self.assertLen(self.fake_process.sigma_arg_history, 2)


class RFScheduleTest(absltest.TestCase):

  def test_alpha_and_sigma(self):

    self.assertEqual(schedules.RFSchedule().alpha(jnp.array([0.4])), 0.6)
    self.assertEqual(schedules.RFSchedule().sigma(jnp.array([0.4])), 0.4)


class CosineScheduleTest(absltest.TestCase):

  def test_alpha_and_sigma(self):

    self.assertEqual(
        schedules.CosineSchedule().alpha(jnp.array([0.4])), 0.809017
    )
    self.assertEqual(
        schedules.CosineSchedule().sigma(jnp.array([0.4])), 0.58778524
    )


class LinearDiscreteScheduleTest(absltest.TestCase):

  def test_alpha(self):

    self.assertEqual(
        schedules.LinearDiscreteSchedule().alpha(jnp.array([0.4])), 0.6
    )


class CosineDiscreteScheduleTest(absltest.TestCase):

  def test_alpha(self):

    self.assertEqual(
        schedules.CosineDiscreteSchedule().alpha(jnp.array([0.4])), 0.809017
    )


class SquareCosineDiscreteScheduleTest(absltest.TestCase):

  def test_alpha(self):

    self.assertEqual(
        schedules.SquareCosineDiscreteSchedule(s=0.08).alpha(jnp.array([0.4])),
        0.5948411,
    )


SCHEDULES = [
    schedules.RFSchedule(),
    schedules.CosineSchedule(),
    schedules.InverseCosineSchedule(),
    schedules.LinearDiffusionSchedule(),
    schedules.GeometricSchedule(sigma_min=0.1, sigma_max=0.9),
]


class NumericalGaussianScheduleTest(parameterized.TestCase):

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_alpha_sigma_bounds(self, schedule):
    t = jnp.linspace(0.01, 0.99, 100)
    # check alpha and sigma are in [0, 1]
    alpha = schedule.alpha(t)
    self.assertTrue(jnp.all(alpha >= 0.0))
    self.assertTrue(jnp.all(alpha <= 1.0))
    sigma = schedule.sigma(t)
    self.assertTrue(jnp.all(sigma >= 0.0))
    self.assertTrue(jnp.all(sigma <= 1.0))

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_logsnr_f_and_g(self, schedule):
    # check that logsnr is approximately correct.
    t = jnp.linspace(0.01, 0.99, 100)
    logsnr = schedule.logsnr(t)
    alpha = schedule.alpha(t)
    sigma = schedule.sigma(t)
    with self.subTest(func_name='logsnr'):
      logsnr_expected = 2.0 * (jnp.log(alpha) - jnp.log(sigma))
      self.assertTrue(
          jnp.allclose(logsnr, logsnr_expected, atol=1e-6, rtol=1e-6),
          'logsnr() check failed: Absolute difference:'
          f' {jnp.max(jnp.abs(logsnr - logsnr_expected))}',
      )

    # check that f is approximately correct.
    with self.subTest(func_name='f'):
      f = schedule.f(t)
      f_expected = utils.egrad(schedule.alpha)(t) / schedule.alpha(t)
      self.assertTrue(
          jnp.allclose(f, f_expected, atol=1e-6, rtol=1e-6),
          'f() check failed: Absolute difference:'
          f' {jnp.max(jnp.abs(f - f_expected))}',
      )

    # check that g is approximately correct.
    with self.subTest(func_name='g'):
      g = schedule.g(t)
      g_expected = schedule.sigma(t) * jnp.sqrt(
          -utils.egrad(schedule.logsnr)(t)
      )
      self.assertTrue(
          jnp.allclose(g, g_expected, atol=1e-6, rtol=1e-6),
          'g() check failed: Absolute difference:'
          f' {jnp.max(jnp.abs(g - g_expected))}',
      )

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_custom_gradients(self, schedule):
    t = jnp.linspace(0.01, 0.99, 100)
    for func_name in ('alpha', 'sigma', 'logsnr'):
      # Use __dict__ to avoid triggering descriptor
      custom_grad = schedule.__class__.__dict__.get(func_name, None)
      if custom_grad and isinstance(custom_grad, utils.CustomGradient):
        # This means the function has a @utils.CustomGradient
        with self.subTest(func_name=func_name):
          fn = getattr(schedule, func_name)
          orig_fn = functools.partial(custom_grad.primal_fn, schedule)
          # This triggers the custom gradient implementation
          custom_grad = utils.egrad(fn)(t)
          # This circumvents the custom gradient implementation
          jax_grad = utils.egrad(orig_fn)(t)
          self.assertTrue(
              jnp.allclose(custom_grad, jax_grad, atol=1e-6, rtol=1e-6),
              f'{func_name}() custom gradientcheck failed: Absolute difference:'
              f' {jnp.max(jnp.abs(custom_grad - jax_grad))}',
          )

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_inverse_logsnr(self, schedule):
    t = jnp.linspace(0.01, 0.99, 100)
    self.assertTrue(
        jnp.allclose(t, _identity_logsnr(schedule, t), atol=1e-6, rtol=1e-6),
        'inverse_logsnr() check failed: Absolute difference:'
        f' {jnp.max(jnp.abs(t - _identity_logsnr(schedule, t)))}',
    )

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_time_change_shift(self, schedule):
    shifted_schedule = schedules.ShiftedSchedule(
        original_schedule=schedule,
        target_resolution=1024,
        base_resolution=256,
    )

    t = jnp.linspace(0.01, 0.99, 100)
    self.assertTrue(
        jnp.allclose(
            t, _identity_time(shifted_schedule, t), atol=1e-5, rtol=1e-5
        ),
        'inverse_logsnr() check failed: Absolute difference:'
        f' {jnp.max(jnp.abs(t - _identity_time(shifted_schedule, t)))}',
    )
    # 1e-6 too strong for InverseCosineSchedule

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_inverse_logsnr_shift(self, schedule):
    shifted_schedule = schedules.ShiftedSchedule(
        original_schedule=schedule,
        target_resolution=1024,
        base_resolution=256,
    )

    t = jnp.linspace(0.01, 0.99, 100)
    self.assertTrue(
        jnp.allclose(
            t, _identity_logsnr(shifted_schedule, t), atol=1e-5, rtol=1e-5
        ),
        'inverse_logsnr() check failed: Absolute difference:'
        f' {jnp.max(jnp.abs(t - _identity_logsnr(shifted_schedule, t)))}',
    )
    # 1e-6 too strong for InverseCosineSchedule

  @parameterized.named_parameters(*(
      {
          'testcase_name': schedule.__class__.__name__,
          'schedule': schedule,
      }
      for schedule in SCHEDULES
  ))
  def test_shifted_endpoints(self, schedule):
    max_logsnr_original = 3.0
    min_logsnr_original = -3.0

    shifted_schedule = schedules.ShiftedSchedule(
        original_schedule=schedule,
        target_resolution=128,
        base_resolution=32,
        logsnr_max=max_logsnr_original,
        logsnr_min=min_logsnr_original,
    )

    t_min, t_max = jnp.array([0.0]), jnp.array([1.0])

    max_logsnr = shifted_schedule.logsnr(t_min)
    min_logsnr = shifted_schedule.logsnr(t_max)

    self.assertAlmostEqual(
        max_logsnr,
        max_logsnr_original + shifted_schedule.logsnr_shift,
        places=5,
        msg='LogSNR at t=0.0 of the shifted schedule is not correct.',
    )
    self.assertAlmostEqual(
        min_logsnr,
        min_logsnr_original + shifted_schedule.logsnr_shift,
        places=5,
        msg='LogSNR at t=1.0 of the shifted schedule is not correct.',
    )

  def test_shifted_cosine_schedule(self):
    max_logsnr_original = 3.0
    min_logsnr_original = -3.0

    original_schedule = schedules.CosineSchedule()

    shifted_schedule = schedules.ShiftedSchedule(
        original_schedule=original_schedule,
        target_resolution=128,
        base_resolution=32,
        logsnr_max=max_logsnr_original,
        logsnr_min=min_logsnr_original,
    )

    t = jnp.linspace(0.0, 1.0, 100)
    alphas = shifted_schedule.alpha(t)
    sigmas = shifted_schedule.sigma(t)

    rescaled_time = (
        t * (shifted_schedule.tmax - shifted_schedule.tmin)
        + shifted_schedule.tmin
    )
    original_log_snr = original_schedule.logsnr(rescaled_time)
    shifted_log_snr = original_log_snr + shifted_schedule.logsnr_shift
    alphas_expected = jnp.sqrt(jax.nn.sigmoid(shifted_log_snr))
    sigmas_expected = jnp.sqrt(jax.nn.sigmoid(-shifted_log_snr))

    self.assertTrue(
        jnp.allclose(alphas, alphas_expected, atol=1e-6, rtol=1e-6),
        'alpha() check failed: Absolute difference:'
        f' {jnp.max(jnp.abs(alphas - alphas_expected))}',
    )
    self.assertTrue(
        jnp.allclose(sigmas, sigmas_expected, atol=1e-6, rtol=1e-6),
        'sigma() check failed: Absolute difference:'
        f' {jnp.max(jnp.abs(sigmas - sigmas_expected))}',
    )


if __name__ == '__main__':
  absltest.main()
