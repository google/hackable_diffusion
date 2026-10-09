# Copyright 2026 Hackable Diffusion Authors.
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

import dataclasses
import functools
import itertools

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import gaussian
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


TimeArray = hd_typing.TimeArray

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

    schedule = gaussian.CosineSchedule()

    key_xt, key_initial_val = jax.random.split(jax.random.PRNGKey(42))
    xt = jax.random.uniform(key_xt, shape, minval=-1.0, maxval=1.0)
    initial_val = jax.random.uniform(
        key_initial_val, shape, minval=-1.0, maxval=1.0
    )

    sigma = schedule.sigma(t)
    alpha = schedule.alpha(t)
    alpha_der = jax_helpers.egrad(schedule.alpha)(t)
    sigma_der = jax_helpers.egrad(schedule.sigma)(t)

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


################################################################################
# MARK: Helper functions
################################################################################


def _identity_logsnr(schedule: gaussian.GaussianSchedule, x: TimeArray):
  return schedule.logsnr(schedule.inverse_logsnr(x))


def _identity_time(schedule: gaussian.ShiftedSchedule, x: TimeArray):
  return schedule.time_change(schedule.inverse_time_change(x))


################################################################################
# MARK: Schedule Tests
################################################################################


class GaussianNoiseScheduleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    @dataclasses.dataclass(frozen=True)
    class DummyNoiseSchedule(gaussian.GaussianSchedule):

      alpha_arg_history, sigma_arg_history = [], []

      def alpha(self, t):
        self.alpha_arg_history.append(t)
        return 1.0 - t

      def sigma(self, t):
        self.sigma_arg_history.append(t)
        return t

    self.fake_process = DummyNoiseSchedule()  # pyrefly: ignore[bad-instantiation]

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

    self.assertEqual(gaussian.RFSchedule().alpha(jnp.array([0.4])), 0.6)
    self.assertEqual(gaussian.RFSchedule().sigma(jnp.array([0.4])), 0.4)


class CosineScheduleTest(absltest.TestCase):

  def test_alpha_and_sigma(self):

    self.assertEqual(
        gaussian.CosineSchedule().alpha(jnp.array([0.4])), 0.809017
    )
    self.assertEqual(
        gaussian.CosineSchedule().sigma(jnp.array([0.4])), 0.58778524
    )


SCHEDULES = [
    gaussian.RFSchedule(),
    gaussian.CosineSchedule(),
    gaussian.InverseCosineSchedule(),
    gaussian.LinearDiffusionSchedule(),
    gaussian.GeometricSchedule(sigma_min=0.1, sigma_max=0.9),
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
      f_expected = jax_helpers.egrad(schedule.alpha)(t) / schedule.alpha(t)
      self.assertTrue(
          jnp.allclose(f, f_expected, atol=1e-6, rtol=1e-6),
          'f() check failed: Absolute difference:'
          f' {jnp.max(jnp.abs(f - f_expected))}',
      )

    # check that g is approximately correct.
    with self.subTest(func_name='g'):
      g = schedule.g(t)
      g_expected = schedule.sigma(t) * jnp.sqrt(
          -jax_helpers.egrad(schedule.logsnr)(t)
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
      if custom_grad and isinstance(custom_grad, jax_helpers.CustomGradient):
        # This means the function has a @jax_helpers.CustomGradient
        with self.subTest(func_name=func_name):
          fn = getattr(schedule, func_name)
          orig_fn = functools.partial(custom_grad.primal_fn, schedule)
          # This triggers the custom gradient implementation
          custom_grad = jax_helpers.egrad(fn)(t)
          # This circumvents the custom gradient implementation
          jax_grad = jax_helpers.egrad(orig_fn)(t)
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
    shifted_schedule = gaussian.ShiftedSchedule(
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
    shifted_schedule = gaussian.ShiftedSchedule(
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

    shifted_schedule = gaussian.ShiftedSchedule(
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

    original_schedule = gaussian.CosineSchedule()

    shifted_schedule = gaussian.ShiftedSchedule(
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

  def test_clamped_schedule(self):
    max_logsnr = 3.0
    min_logsnr = -3.0
    original_schedule = gaussian.CosineSchedule()
    clamped_schedule = gaussian.ClampedSchedule(
        original_schedule=original_schedule,
        logsnr_max=max_logsnr,
        logsnr_min=min_logsnr,
    )

    t_0, t_1 = jnp.array([0.0]), jnp.array([1.0])
    self.assertAlmostEqual(clamped_schedule.logsnr(t_0), max_logsnr, places=5)
    self.assertAlmostEqual(clamped_schedule.logsnr(t_1), min_logsnr, places=5)

    t = jnp.linspace(0.0, 1.0, 50)
    recovered_t = clamped_schedule.inverse_logsnr(clamped_schedule.logsnr(t))
    self.assertTrue(jnp.allclose(t, recovered_t, atol=1e-5, rtol=1e-5))

    alphas = clamped_schedule.alpha(t)
    sigmas = clamped_schedule.sigma(t)
    self.assertTrue(jnp.allclose(alphas**2 + sigmas**2, 1.0, atol=1e-5))

  def test_offset_schedule(self):
    original_schedule = gaussian.RFSchedule()
    offset_val = 1.5
    offset_schedule = gaussian.OffsetSchedule(
        original_schedule=original_schedule,
        offset=offset_val,
    )

    t = jnp.linspace(0.1, 0.9, 50)
    expected_logsnr = original_schedule.logsnr(t) + offset_val
    self.assertTrue(
        jnp.allclose(offset_schedule.logsnr(t), expected_logsnr, atol=1e-5)
    )

    recovered_t = offset_schedule.inverse_logsnr(offset_schedule.logsnr(t))
    self.assertTrue(jnp.allclose(t, recovered_t, atol=1e-5))

    # Test resolution-based offset
    res_schedule = gaussian.OffsetSchedule(
        original_schedule=original_schedule,
        target_resolution=128,
        base_resolution=32,
    )
    expected_offset = -2.0 * (jnp.log(128) - jnp.log(32))
    self.assertAlmostEqual(res_schedule.logsnr_offset, expected_offset, places=5)

  def test_shifted_schedule_composition_equivalence(self):
    original_schedule = gaussian.CosineSchedule()
    target_res = 128
    base_res = 32
    max_logsnr = 4.0
    min_logsnr = -4.0

    shifted_schedule = gaussian.ShiftedSchedule(
        original_schedule=original_schedule,
        target_resolution=target_res,
        base_resolution=base_res,
        logsnr_max=max_logsnr,
        logsnr_min=min_logsnr,
    )

    clamped = gaussian.ClampedSchedule(
        original_schedule=original_schedule,
        logsnr_max=max_logsnr,
        logsnr_min=min_logsnr,
    )
    offset_sched = gaussian.OffsetSchedule(
        original_schedule=clamped,
        offset=shifted_schedule.logsnr_shift,
    )

    t = jnp.linspace(0.01, 0.99, 50)
    self.assertTrue(
        jnp.allclose(
            shifted_schedule.logsnr(t), offset_sched.logsnr(t), atol=1e-6
        )
    )
    self.assertTrue(
        jnp.allclose(
            shifted_schedule.alpha(t), offset_sched.alpha(t), atol=1e-6
        )
    )
    self.assertTrue(
        jnp.allclose(
            shifted_schedule.sigma(t), offset_sched.sigma(t), atol=1e-6
        )
    )

    logsnrs = shifted_schedule.logsnr(t)
    self.assertTrue(
        jnp.allclose(
            shifted_schedule.inverse_logsnr(logsnrs),
            offset_sched.inverse_logsnr(logsnrs),
            atol=1e-6,
        )
    )

  def test_blend_schedule(self):
    original_schedule = gaussian.CosineSchedule()
    clamped = gaussian.ClampedSchedule(
        original_schedule=original_schedule,
        logsnr_max=12.0,
        logsnr_min=-12.0,
    )
    shifted = gaussian.OffsetSchedule(
        original_schedule=clamped,
        target_resolution=64,
        base_resolution=32,
    )
    blend = gaussian.BlendSchedule(
        schedule_0=clamped,
        schedule_1=shifted,
    )

    # At t=0: logsnr is unshifted (12.0)
    t_0 = jnp.array([0.0])
    self.assertAlmostEqual(float(blend.logsnr(t_0)[0]), 12.0, places=4)

    # At t=1: logsnr is fully shifted (-12.0 - 2*ln(2))
    t_1 = jnp.array([1.0])
    expected_logsnr_1 = -12.0 - 2.0 * float(jnp.log(2.0))
    self.assertAlmostEqual(
        float(blend.logsnr(t_1)[0]), expected_logsnr_1, places=4
    )

    # Midpoints: verify (1-t)*s0 + t*s1 == s0 + t*offset
    t_mid = jnp.array([0.1, 0.5, 0.9])
    expected_mid = clamped.logsnr(t_mid) + t_mid * shifted.logsnr_offset
    self.assertTrue(jnp.allclose(blend.logsnr(t_mid), expected_mid, atol=1e-5))

    # Alpha and sigma properties: alpha^2 + sigma^2 == 1
    alpha = blend.alpha(t_mid)
    sigma = blend.sigma(t_mid)
    self.assertTrue(jnp.allclose(alpha**2 + sigma**2, 1.0, atol=1e-5))

    # Inverse logsnr roundtrip via Newton-Raphson
    t_rec = blend.inverse_logsnr(blend.logsnr(t_mid))
    self.assertTrue(jnp.allclose(t_rec, t_mid, atol=1e-4))


if __name__ == '__main__':
  absltest.main()
