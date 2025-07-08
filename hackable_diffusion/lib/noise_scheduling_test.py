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

from hackable_diffusion.lib import noise_scheduling

import jax

from absl.testing import absltest


class NoiseScheduleTest(absltest.TestCase):

  def test_manual_gradient_definition(self):

    def value_fn(x):
      return x**2

    def derivative_fn(x):
      return 3. * x

    def dumm_fn(x):
      return noise_scheduling._manual_gradient_definition(
          value_fn=value_fn,
          derivative_fn=derivative_fn,
          x=x,
          )

    self.assertEqual(dumm_fn(1.), 1.)
    self.assertEqual(jax.grad(dumm_fn)(1.), 3.)
    self.assertEqual(jax.grad(value_fn)(1.), 2.)


class GaussianNoiseScheduleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    @dataclasses.dataclass(frozen=True)
    class DummyNoiseSchedule(noise_scheduling.GaussianNoiseSchedule):

      alpha_arg_history, sigma_arg_history = [], []

      def alpha(self, t):
        self.alpha_arg_history.append(t)
        return 1. - t

      def sigma(self, t):
        self.sigma_arg_history.append(t)
        return t

    self.fake_process = DummyNoiseSchedule()

  def test_logsnr(self):
    _ = self.fake_process.logsnr(1.)

    self.assertLen(self.fake_process.alpha_arg_history, 1)
    self.assertLen(self.fake_process.sigma_arg_history, 1)

  def test_grad_logsnr(self):
    _ = jax.grad(self.fake_process.logsnr)(1.)

    self.assertLen(self.fake_process.alpha_arg_history, 3)
    self.assertLen(self.fake_process.sigma_arg_history, 3)

  def test_f(self):
    _ = self.fake_process.f(1.)

    self.assertLen(self.fake_process.alpha_arg_history, 2)
    self.assertEmpty(self.fake_process.sigma_arg_history)

  def test_g(self):
    _ = self.fake_process.g(1.)

    self.assertLen(self.fake_process.alpha_arg_history, 3)
    self.assertLen(self.fake_process.sigma_arg_history, 4)


class RFScheduleTest(absltest.TestCase):

  def test_alpha_and_sigma(self):

    self.assertEqual(noise_scheduling.RFSchedule().alpha(.4), 0.6)
    self.assertEqual(noise_scheduling.RFSchedule().sigma(.4), 0.4)


class CosineScheduleTest(absltest.TestCase):

  def test_alpha_and_sigma(self):

    self.assertEqual(noise_scheduling.CosineSchedule().alpha(.4), 0.809017)
    self.assertEqual(noise_scheduling.CosineSchedule().sigma(.4), 0.58778524)


if __name__ == '__main__':
  absltest.main()
