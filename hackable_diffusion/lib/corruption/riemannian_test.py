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

"""Tests for Riemannian Flow Matching corruption process."""

from absl.testing import absltest
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption import riemannian
import jax
import jax.numpy as jnp
import numpy as np


def _make_process(manifold):
  return riemannian.RiemannianProcess(
      manifold=manifold,
      schedule=riemannian.LinearRiemannianSchedule(),
  )


class SphereCorruptionTest(absltest.TestCase):

  def test_corrupt(self):
    manifold = manifolds.Sphere()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(0)

    batch_size = 8
    x0 = manifold.random_uniform(key, (batch_size, 3))
    time = jnp.linspace(0, 1, batch_size)

    xt, target_info = process.corrupt(key, x0, time)

    # xt should be on the sphere.
    norms = jnp.linalg.norm(xt, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Velocity should be tangent to the sphere at xt, i.e. <xt, vel> = 0.
    vel = target_info['velocity']
    self.assertEqual(vel.shape, (batch_size, 3))
    inner_products = jnp.sum(xt * vel, axis=-1)
    np.testing.assert_allclose(inner_products, 0.0, atol=1e-5)

  def test_velocity_at_t1(self):
    """At t=1, alpha=0 so xt = x1 and velocity = -log(x1, x0)."""
    manifold = manifolds.Sphere()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(0)

    x0 = jnp.array([[1.0, 0.0, 0.0]])
    t1 = jnp.array([1.0])
    xt1, target1 = process.corrupt(key, x0, t1)
    x1_sampled = target1['x1']
    # At t=1, alpha=0: geodesic(x1, x0, 0) = x1.
    np.testing.assert_allclose(xt1, x1_sampled, atol=1e-5)

    # velocity = alpha_dot(1) * velocity(x1, x0, 0) = -1 * log(x1, x0).
    v1 = target1['velocity']
    v_log = manifold.log(x1_sampled, x0)
    np.testing.assert_allclose(v1, -v_log, atol=1e-5)

  def test_identity_at_t0(self):
    """At t=0, alpha=1 so xt = x0 (identity corruption)."""
    manifold = manifolds.Sphere()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(0)

    x0 = jnp.array([[1.0, 0.0, 0.0]])
    t0 = jnp.array([0.0])
    xt0, _ = process.corrupt(key, x0, t0)
    # At t=0, alpha=1: geodesic(x1, x0, 1) = x0.
    np.testing.assert_allclose(xt0, x0, atol=1e-5)


class SO3CorruptionTest(absltest.TestCase):

  def test_corrupt(self):
    manifold = manifolds.SO3()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(1)

    batch_size = 8
    x0 = manifold.random_uniform(key, (batch_size, 3, 3))
    time = jnp.linspace(0, 1, batch_size)

    xt, target_info = process.corrupt(key, x0, time)

    # xt should be a valid rotation: R^T R = I and det(R) = 1.
    rtrt = jnp.matmul(jnp.swapaxes(xt, -2, -1), xt)
    eyes = jnp.broadcast_to(jnp.eye(3), rtrt.shape)
    np.testing.assert_allclose(rtrt, eyes, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(xt), 1.0, atol=1e-5)

    # Velocity should be in the tangent space: x^T v is skew-symmetric.
    vel = target_info['velocity']
    self.assertEqual(vel.shape, (batch_size, 3, 3))


class LinearRiemannianScheduleTest(absltest.TestCase):

  def test_alpha_boundary_values(self):
    schedule = riemannian.LinearRiemannianSchedule()
    # At t=0, alpha(0) = 1.0 (data).
    self.assertAlmostEqual(
        schedule.alpha(jnp.array([0.0])).item(), 1.0, places=6
    )
    # At t=1, alpha(1) = 0.0 (noise / base distribution).
    self.assertAlmostEqual(
        schedule.alpha(jnp.array([1.0])).item(), 0.0, places=6
    )

  def test_alpha_intermediate(self):
    schedule = riemannian.LinearRiemannianSchedule()
    self.assertAlmostEqual(
        schedule.alpha(jnp.array([0.4])).item(), 0.6, places=6
    )

  def test_alpha_dot(self):
    schedule = riemannian.LinearRiemannianSchedule()
    t = jnp.linspace(0.01, 0.99, 50)
    alpha_dot = schedule.alpha_dot(t)
    # For linear schedule, alpha_dot(t) = -1.0 for all t.
    self.assertTrue(jnp.allclose(alpha_dot, -jnp.ones_like(t), atol=1e-6))

  def test_alpha_dot_matches_autodiff(self):
    schedule = riemannian.LinearRiemannianSchedule()
    t = jnp.linspace(0.01, 0.99, 50)
    alpha_dot = schedule.alpha_dot(t)
    alpha_dot_auto = jax_helpers.egrad(schedule.alpha)(t)
    self.assertTrue(
        jnp.allclose(alpha_dot, alpha_dot_auto, atol=1e-6, rtol=1e-6),
        'alpha_dot() does not match autodiff: Absolute difference:'
        f' {jnp.max(jnp.abs(alpha_dot - alpha_dot_auto))}',
    )

  def test_alpha_bounds(self):
    schedule = riemannian.LinearRiemannianSchedule()
    t = jnp.linspace(0.0, 1.0, 100)
    alpha = schedule.alpha(t)
    self.assertTrue(jnp.all(alpha >= 0.0))
    self.assertTrue(jnp.all(alpha <= 1.0))

  def test_evaluate(self):
    schedule = riemannian.LinearRiemannianSchedule()
    t = jnp.array([0.3])
    result = schedule.evaluate(t)
    self.assertIn('time', result)
    self.assertIn('alpha', result)
    self.assertIn('alpha_dot', result)
    self.assertAlmostEqual(result['alpha'].item(), 0.7, places=6)
    self.assertAlmostEqual(result['alpha_dot'].item(), -1.0, places=6)

  def test_velocity_at_t1(self):
    """At t=1, alpha=0 so xt = x1 and velocity = -log(x1, x0)."""
    manifold = manifolds.SO3()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(1)

    x0 = jnp.eye(3)[None, ...]  # (1, 3, 3)
    t1 = jnp.array([1.0])
    xt1, target1 = process.corrupt(key, x0, t1)
    x1_sampled = target1['x1']
    # At t=1, alpha=0: geodesic(x1, x0, 0) = x1.
    np.testing.assert_allclose(xt1, x1_sampled, atol=1e-5)

    # velocity = alpha_dot(1) * velocity(x1, x0, 0) = -1 * log(x1, x0).
    v1 = target1['velocity']
    v_log = manifold.log(x1_sampled, x0)
    np.testing.assert_allclose(v1, -v_log, atol=1e-4)

  def test_identity_at_t0(self):
    """At t=0, alpha=1 so xt = x0 (identity corruption)."""
    manifold = manifolds.SO3()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(1)

    x0 = jnp.eye(3)[None, ...]  # (1, 3, 3)
    t0 = jnp.array([0.0])
    xt0, _ = process.corrupt(key, x0, t0)
    # At t=0, alpha=1: geodesic(x1, x0, 1) = x0.
    np.testing.assert_allclose(xt0, x0, atol=1e-5)


class TorusCorruptionTest(absltest.TestCase):

  def test_corrupt(self):
    manifold = manifolds.Torus()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(2)

    batch_size = 8
    dim = 4
    x0 = manifold.random_uniform(key, (batch_size, dim))
    time = jnp.linspace(0, 1, batch_size)

    xt, target_info = process.corrupt(key, x0, time)

    # xt should be in [0, 1).
    self.assertTrue(jnp.all(xt >= 0.0))
    self.assertTrue(jnp.all(xt < 1.0))

    vel = target_info['velocity']
    self.assertEqual(vel.shape, (batch_size, dim))

  def test_velocity_at_t1(self):
    """At t=1, alpha=0 so xt = x1 and velocity = -log(x1, x0)."""
    manifold = manifolds.Torus()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(2)

    x0 = jnp.array([[0.1, 0.5, 0.9]])
    t1 = jnp.array([1.0])
    xt1, target1 = process.corrupt(key, x0, t1)
    x1_sampled = target1['x1']
    # At t=1, alpha=0: geodesic(x1, x0, 0) = x1.
    np.testing.assert_allclose(xt1, x1_sampled, atol=1e-5)

    # velocity = alpha_dot(1) * velocity(x1, x0, 0) = -1 * log(x1, x0).
    v1 = target1['velocity']
    v_log = manifold.log(x1_sampled, x0)
    np.testing.assert_allclose(v1, -v_log, atol=1e-5)

  def test_identity_at_t0(self):
    """At t=0, alpha=1 so xt = x0 (identity corruption)."""
    manifold = manifolds.Torus()
    process = _make_process(manifold)
    key = jax.random.PRNGKey(2)

    x0 = jnp.array([[0.1, 0.5, 0.9]])
    t0 = jnp.array([0.0])
    xt0, _ = process.corrupt(key, x0, t0)
    # At t=0, alpha=1: geodesic(x1, x0, 1) = x0.
    np.testing.assert_allclose(xt0, x0, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
