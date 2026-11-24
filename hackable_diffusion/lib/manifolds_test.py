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

"""Unit tests for Sphere, SO(3), and Torus Riemannian manifolds."""

from absl.testing import absltest
from absl.testing import parameterized
from hackable_diffusion.lib import manifolds
import jax
import jax.numpy as jnp
import numpy as np

################################################################################
# MARK: Helper functions
################################################################################


def _random_so3_tangent(key, manifold, x, scale=0.5):
  """Generate a random tangent vector at R on SO(3) with bounded norm."""
  v = scale * jax.random.normal(key, x.shape)
  return manifold.project(x, v)


def _geodesic(manifold, x, y, t):
  """Geodesic between x and y at time t in [0, 1]."""
  return manifold.exp(x, t * manifold.log(x, y))


def _dist_sq(manifold, x, y):
  """Squared Riemannian distance."""
  return jnp.square(manifold.dist(x, y))


################################################################################
# MARK: Sphere tests
################################################################################


class SphereTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.manifold = manifolds.Sphere()

  def test_exp_log_roundtrip(self):
    key = jax.random.PRNGKey(0)
    x = self.manifold.random_uniform(key, (10, 3))
    v = jax.random.normal(key, (10, 3))
    v = self.manifold.project(x, v)

    y = self.manifold.exp(x, v)
    v_rec = self.manifold.log(x, y)

    np.testing.assert_allclose(v, v_rec, atol=1e-5)
    # Check output stays on sphere.
    np.testing.assert_allclose(jnp.linalg.norm(y, axis=-1), 1.0, atol=1e-5)

  def test_velocity_matches_numerical_derivative(self):
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (1, 3))
    y = self.manifold.random_uniform(k2, (1, 3))
    t = jnp.array([[0.5]])

    v_analytic = self.manifold.velocity(x=x, y=y, t=t)

    # Numerical derivative via central differences.
    eps = 1e-4
    xt_plus = _geodesic(self.manifold, x, y, t + eps)
    xt_minus = _geodesic(self.manifold, x, y, t - eps)
    v_num = (xt_plus - xt_minus) / (2 * eps)

    np.testing.assert_allclose(v_analytic, v_num, atol=1e-3)

  def test_dist(self):
    """Test shape and positivity of distance."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (5, 4))
    y = self.manifold.random_uniform(k2, (5, 4))

    d = self.manifold.dist(x, y)
    self.assertEqual(d.shape, (5,))
    # Distance should be non-negative.
    self.assertTrue(jnp.all(d >= 0.0))

  def test_dist_self(self):
    """Distance to self should be ~zero."""
    key = jax.random.PRNGKey(42)
    x = self.manifold.random_uniform(key, (5, 4))
    # Not exactly zero due to epsilon clipping
    # in arccos: dist(x,x) = arccos(clip(<x,x>)) ≈ arccos(1-eps) > 0.
    d_self = self.manifold.dist(x, x)
    np.testing.assert_allclose(d_self, 0.0, atol=2e-3)

  def test_dist_known_value(self):
    """Distance between orthogonal unit vectors should be pi/2."""
    x = jnp.array([[1.0, 0.0, 0.0]])
    y = jnp.array([[0.0, 1.0, 0.0]])
    d = self.manifold.dist(x, y)
    np.testing.assert_allclose(d, jnp.array([jnp.pi / 2]), atol=1e-5)

  def test_dist_sq(self):
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (5, 3))
    y = self.manifold.random_uniform(k2, (5, 3))

    d = self.manifold.dist(x, y)
    d_sq = _dist_sq(self.manifold, x, y)
    np.testing.assert_allclose(d_sq, d**2, atol=1e-6)

  def test_geodesic_endpoints(self):
    """Geodesic at t=0 gives x, at t=1 gives y."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (3, 5))
    y = self.manifold.random_uniform(k2, (3, 5))

    t0 = jnp.zeros((3, 1))
    t1 = jnp.ones((3, 1))

    g0 = _geodesic(self.manifold, x, y, t0)
    g1 = _geodesic(self.manifold, x, y, t1)

    np.testing.assert_allclose(g0, x, atol=1e-5)
    np.testing.assert_allclose(g1, y, atol=1e-5)

  @parameterized.parameters(0.0, 0.25, 0.5, 0.75, 1.0)
  def test_geodesic_stays_on_manifold(self, t_val):
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (5, 3))
    y = self.manifold.random_uniform(k2, (5, 3))

    t = jnp.full((5, 1), t_val)
    gt = _geodesic(self.manifold, x, y, t)
    norms = jnp.linalg.norm(gt, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

  def test_exp_zero_tangent(self):
    """exp(x, 0) should return x."""
    key = jax.random.PRNGKey(3)
    x = self.manifold.random_uniform(key, (5, 3))
    v = jnp.zeros_like(x)
    y = self.manifold.exp(x, v)
    np.testing.assert_allclose(y, x, atol=1e-6)

  def test_near_antipodal_log_returns_finite(self):
    """Log map on nearly antipodal points should return finite values."""
    x = jnp.array([[1.0, 0.0, 0.0]])
    y = jnp.array([[-1.0 + 1e-6, 1e-7, 0.0]])
    y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)

    v = self.manifold.log(x, y)
    self.assertTrue(jnp.all(jnp.isfinite(v)))

  def test_dist_consistency_with_log_norm(self):
    """dist(x, y) should equal |log(x, y)|."""
    key = jax.random.PRNGKey(5)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (10, 3))
    y = self.manifold.random_uniform(k2, (10, 3))

    d = self.manifold.dist(x, y)
    v = self.manifold.log(x, y)
    v_norm = jnp.linalg.norm(v, axis=-1)
    np.testing.assert_allclose(d, v_norm, atol=1e-5)

  def test_exp_known_value(self):
    """exp from north pole along e2 by pi/2 should land on equator."""
    # x = north pole (0, 0, 1), v = (0, pi/2, 0) is tangent at x.
    x = jnp.array([[0.0, 0.0, 1.0]])
    v = jnp.array([[0.0, jnp.pi / 2, 0.0]])
    y = self.manifold.exp(x, v)
    # exp(x, v) = cos(|v|)*x + sinc(|v|)*v
    #           = cos(pi/2)*(0,0,1) + sin(pi/2)/|v| * v
    #           = 0*(0,0,1) + 1*(0,1,0) = (0,1,0)
    expected = jnp.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(y, expected, atol=1e-5)

  def test_log_known_value(self):
    """log between two orthogonal unit vectors gives tangent with norm pi/2."""
    x = jnp.array([[1.0, 0.0, 0.0]])
    y = jnp.array([[0.0, 1.0, 0.0]])
    v = self.manifold.log(x, y)
    # The tangent vector should point from x toward y in the tangent plane,
    # which is the e2 direction, with magnitude pi/2 (the geodesic angle).
    expected = jnp.array([[0.0, jnp.pi / 2, 0.0]])
    np.testing.assert_allclose(v, expected, atol=1e-5)

  def test_velocity_asymmetry(self):
    """velocity(x, y, t) should differ from velocity(y, x, t)."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (1, 3))
    y = self.manifold.random_uniform(k2, (1, 3))
    t = jnp.array([[0.5]])

    v_xy = self.manifold.velocity(x=x, y=y, t=t)
    v_yx = self.manifold.velocity(x=y, y=x, t=t)

    self.assertFalse(jnp.allclose(v_xy, v_yx, atol=1e-3))


################################################################################
# MARK: SO(3) tests
################################################################################


class SO3Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.manifold = manifolds.SO3()

  def test_exp_log_roundtrip(self):
    key = jax.random.PRNGKey(0)
    x = self.manifold.random_uniform(key, (10, 3, 3))
    v = _random_so3_tangent(key, self.manifold, x)

    p = self.manifold.exp(x, v)
    v_rec = self.manifold.log(x, p)

    np.testing.assert_allclose(v, v_rec, atol=1e-5)
    # Check orthogonality of result.
    rtr = jnp.matmul(jnp.swapaxes(p, -1, -2), p)
    i_mat = jnp.eye(3)
    np.testing.assert_allclose(
        rtr, jnp.broadcast_to(i_mat, rtr.shape), atol=1e-5
    )

  def test_velocity_matches_numerical_derivative(self):
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    r = self.manifold.random_uniform(k1, (1, 3, 3))
    p = self.manifold.random_uniform(k2, (1, 3, 3))
    t = jnp.array([[0.5]])

    v_analytic = self.manifold.velocity(x=r, y=p, t=t[..., None])

    # Numerical derivative via central differences.
    eps = 1e-4
    xt_plus = _geodesic(self.manifold, r, p, (t + eps)[..., None])
    xt_minus = _geodesic(self.manifold, r, p, (t - eps)[..., None])
    v_num = (xt_plus - xt_minus) / (2 * eps)

    np.testing.assert_allclose(v_analytic, v_num, atol=1e-2)

  def test_dist(self):
    """Test shape and positivity of distance."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    r = self.manifold.random_uniform(k1, (5, 3, 3))
    p = self.manifold.random_uniform(k2, (5, 3, 3))

    d = self.manifold.dist(r, p)
    self.assertEqual(d.shape, (5,))
    self.assertTrue(jnp.all(d >= 0.0))

  def test_dist_self(self):
    """Distance to self should be ~zero."""
    key = jax.random.PRNGKey(42)
    r = self.manifold.random_uniform(key, (5, 3, 3))
    # Not exactly zero due to epsilon clipping.
    d_self = self.manifold.dist(r, r)
    np.testing.assert_allclose(d_self, 0.0, atol=2e-3)

  def test_dist_known_value(self):
    """Distance from identity to Rz(pi/2) should be pi/2."""
    r = jnp.eye(3)[None, ...]  # (1, 3, 3)
    p = jnp.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    d = self.manifold.dist(r, p)
    np.testing.assert_allclose(d, jnp.array([jnp.pi / 2]), atol=1e-5)

  def test_dist_sq(self):
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    r = self.manifold.random_uniform(k1, (5, 3, 3))
    p = self.manifold.random_uniform(k2, (5, 3, 3))

    d = self.manifold.dist(r, p)
    d_sq = _dist_sq(self.manifold, r, p)
    np.testing.assert_allclose(d_sq, d**2, atol=1e-6)

  def test_geodesic_endpoints(self):
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    r = self.manifold.random_uniform(k1, (3, 3, 3))
    p = self.manifold.random_uniform(k2, (3, 3, 3))

    t0 = jnp.zeros((3, 1, 1))
    t1 = jnp.ones((3, 1, 1))

    g0 = _geodesic(self.manifold, r, p, t0)
    g1 = _geodesic(self.manifold, r, p, t1)

    np.testing.assert_allclose(g0, r, atol=1e-5)
    np.testing.assert_allclose(g1, p, atol=1e-4)

  @parameterized.parameters(0.0, 0.25, 0.5, 0.75, 1.0)
  def test_geodesic_stays_on_manifold(self, t_val):
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    r = self.manifold.random_uniform(k1, (5, 3, 3))
    p = self.manifold.random_uniform(k2, (5, 3, 3))
    i_mat = jnp.eye(3)

    t = jnp.full((5, 1, 1), t_val)
    gt = _geodesic(self.manifold, r, p, t)
    rtr = jnp.matmul(jnp.swapaxes(gt, -1, -2), gt)
    np.testing.assert_allclose(
        rtr, jnp.broadcast_to(i_mat, rtr.shape), atol=1e-5
    )

  def test_identity_rotation_log(self):
    """Log of identity rotation from any point should give zero tangent."""
    key = jax.random.PRNGKey(4)
    r = self.manifold.random_uniform(key, (5, 3, 3))
    v = self.manifold.log(r, r)
    np.testing.assert_allclose(v, 0.0, atol=1e-5)

  def test_near_pi_rotation_returns_finite(self):
    """Log map near pi (180 degrees) should return finite values."""
    # Create a rotation matrix that is nearly pi radians from identity.
    angle = jnp.pi - 0.01  # Very close to pi

    # Build rotation matrix via Rodrigues directly.
    k = jnp.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=jnp.float32)
    r_near_pi = jnp.eye(3) + jnp.sin(angle) * k + (1 - jnp.cos(angle)) * k @ k
    r_near_pi = r_near_pi[None, ...]  # Add batch dim: (1, 3, 3)
    r_identity = jnp.eye(3)[None, ...]  # (1, 3, 3)

    v = self.manifold.log(r_identity, r_near_pi)
    self.assertTrue(jnp.all(jnp.isfinite(v)))

  def test_hat_known_value(self):
    """hat maps (x,y,z) to the expected skew-symmetric matrix."""
    v = jnp.array([[1.0, 2.0, 3.0]])
    omega = manifolds._hat(v)
    expected = jnp.array(
        [[[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]]]
    )
    np.testing.assert_allclose(omega, expected, atol=1e-6)

  def test_vee_known_value(self):
    """vee extracts (x,y,z) from a skew-symmetric matrix."""
    omega = jnp.array([[[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]]])
    v = manifolds._vee(omega)
    expected = jnp.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(v, expected, atol=1e-6)

  def test_hat_vee_roundtrip(self):
    """hat(vee(Omega)) should recover Omega for skew-symmetric Omega."""
    key = jax.random.PRNGKey(10)
    v = jax.random.normal(key, (5, 3))
    omega = manifolds._hat(v)
    v_rec = manifolds._vee(omega)
    np.testing.assert_allclose(v, v_rec, atol=1e-6)

  def test_dist_consistency_with_log_norm(self):
    """dist(R, P) should equal |log(R, P)|."""
    key = jax.random.PRNGKey(15)
    k1, k2 = jax.random.split(key)
    r = self.manifold.random_uniform(k1, (10, 3, 3))
    p = self.manifold.random_uniform(k2, (10, 3, 3))

    d = self.manifold.dist(r, p)
    v = self.manifold.log(r, p)
    # Extract the Lie algebra element and compute its norm.
    omega = jnp.matmul(jnp.swapaxes(r, -1, -2), v)
    omega_vec = manifolds._vee(omega)
    v_norm = jnp.linalg.norm(omega_vec, axis=-1)
    np.testing.assert_allclose(d, v_norm, atol=1e-4)

  def test_exp_known_value(self):
    """exp from identity with pi/2 rotation about z-axis gives Rz(pi/2)."""
    r = jnp.eye(3)[None, ...]  # (1, 3, 3)
    # Tangent vector at identity is the skew-symmetric matrix itself
    # For a pi/2 rotation about z: omega_vec = (0, 0, pi/2).
    omega_mat = jnp.array(
        [[[0.0, -jnp.pi / 2, 0.0], [jnp.pi / 2, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    v = omega_mat  # At identity, tangent = R @ Omega = Omega.
    result = self.manifold.exp(r, v)
    expected = jnp.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    np.testing.assert_allclose(result, expected, atol=1e-5)

  def test_log_known_value(self):
    """log from identity to Rz(pi/2) should recover the z-axis generator."""
    r = jnp.eye(3)[None, ...]  # (1, 3, 3)
    p = jnp.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    v = self.manifold.log(r, p)
    # Expected tangent at identity: the skew-symmetric matrix for (0, 0, pi/2).
    expected = jnp.array(
        [[[0.0, -jnp.pi / 2, 0.0], [jnp.pi / 2, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    np.testing.assert_allclose(v, expected, atol=1e-5)

  def test_velocity_asymmetry(self):
    """velocity(x, y, t) should differ from velocity(y, x, t)."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (1, 3, 3))
    y = self.manifold.random_uniform(k2, (1, 3, 3))
    t = jnp.array([[0.5]])

    v_xy = self.manifold.velocity(x=x, y=y, t=t[..., None])
    v_yx = self.manifold.velocity(x=y, y=x, t=t[..., None])

    self.assertFalse(jnp.allclose(v_xy, v_yx, atol=1e-3))


################################################################################
# MARK: Torus tests
################################################################################


class TorusTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.manifold = manifolds.Torus()

  def test_exp_log_roundtrip(self):
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (10, 2))
    v = jax.random.uniform(k2, (10, 2), minval=-0.4, maxval=0.4)

    y = self.manifold.exp(x, v)
    v_rec = self.manifold.log(x, y)

    np.testing.assert_allclose(v, v_rec, atol=1e-5)

  def test_dist(self):
    """Test shape and positivity of distance."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (5, 3))
    y = self.manifold.random_uniform(k2, (5, 3))

    d = self.manifold.dist(x, y)
    self.assertEqual(d.shape, (5,))
    self.assertTrue(jnp.all(d >= 0.0))

  def test_dist_self(self):
    """Distance to self should be zero."""
    key = jax.random.PRNGKey(42)
    x = self.manifold.random_uniform(key, (5, 3))
    d_self = self.manifold.dist(x, x)
    np.testing.assert_allclose(d_self, 0.0, atol=1e-6)

  def test_dist_known_value(self):
    """Distance on torus should use shortest path (wrapping)."""
    x = jnp.array([[0.1, 0.3]])
    y = jnp.array([[0.9, 0.5]])
    d = self.manifold.dist(x, y)
    # Shortest displacements: dim0: 0.9-0.1=0.8 but wrapping gives -0.2,
    # dim1: 0.5-0.3=0.2. So dist = sqrt(0.2^2 + 0.2^2) = 0.2*sqrt(2).
    expected = jnp.array([0.2 * jnp.sqrt(2.0)])
    np.testing.assert_allclose(d, expected, atol=1e-5)

  def test_dist_sq(self):
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (5, 2))
    y = self.manifold.random_uniform(k2, (5, 2))

    d = self.manifold.dist(x, y)
    d_sq = _dist_sq(self.manifold, x, y)
    np.testing.assert_allclose(d_sq, d**2, atol=1e-6)

  def test_geodesic_endpoints(self):
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (3, 2))
    y = self.manifold.random_uniform(k2, (3, 2))

    t0 = jnp.zeros((3, 1))
    t1 = jnp.ones((3, 1))

    g0 = _geodesic(self.manifold, x, y, t0)
    g1 = _geodesic(self.manifold, x, y, t1)

    np.testing.assert_allclose(g0, x, atol=1e-5)
    # g1 should equal y mod 1
    np.testing.assert_allclose(g1 % 1.0, y % 1.0, atol=1e-5)

  def test_velocity_constant(self):
    """Velocity on torus should be constant for all t."""
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (5, 2))
    y = self.manifold.random_uniform(k2, (5, 2))

    v0 = self.manifold.velocity(x=x, y=y, t=jnp.zeros((5, 1)))
    v_half = self.manifold.velocity(x=x, y=y, t=jnp.full((5, 1), 0.5))
    v1 = self.manifold.velocity(x=x, y=y, t=jnp.ones((5, 1)))

    np.testing.assert_allclose(v0, v_half, atol=1e-7)
    np.testing.assert_allclose(v0, v1, atol=1e-7)

  def test_periodic_wrapping(self):
    """Points should wrap around [0, 1)."""
    x = jnp.array([[0.9, 0.1]])
    v = jnp.array([[0.2, -0.3]])
    y = self.manifold.exp(x, v)
    expected = jnp.array([[0.1, 0.8]])
    np.testing.assert_allclose(y, expected, atol=1e-5)

  def test_log_shortest_path(self):
    """Log should return the shortest displacement."""
    x = jnp.array([[0.1]])
    y = jnp.array([[0.9]])
    v = self.manifold.log(x, y)
    # Shortest path wraps around: -0.2, not +0.8.
    np.testing.assert_allclose(v, jnp.array([[-0.2]]), atol=1e-5)

  def test_velocity_asymmetry(self):
    """velocity(x, y, t) should differ from velocity(y, x, t)."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = self.manifold.random_uniform(k1, (1, 2))
    y = self.manifold.random_uniform(k2, (1, 2))
    t = jnp.array([[0.5]])

    v_xy = self.manifold.velocity(x=x, y=y, t=t)
    v_yx = self.manifold.velocity(x=y, y=x, t=t)

    self.assertFalse(jnp.allclose(v_xy, v_yx, atol=1e-3))

  def test_1d_time_broadcast(self):
    key = jax.random.PRNGKey(9)
    k1, k2 = jax.random.split(key)
    t_1d = jnp.array([0.25, 0.75])
    t_3d = t_1d[:, None, None]
    t_4d = t_1d[:, None, None, None]

    # Torus (B=2, N=3, D=4)
    torus = manifolds.Torus()
    tx = torus.random_uniform(k1, (2, 3, 4))
    ty = torus.random_uniform(k2, (2, 3, 4))
    np.testing.assert_allclose(
        manifolds.geodesic(torus, tx, ty, t_1d),
        manifolds.geodesic(torus, tx, ty, t_3d),
        atol=1e-5,
    )

    # SO3 (B=2, N=3, 3, 3)
    so3 = manifolds.SO3()
    rx = so3.random_uniform(k1, (2, 3, 3, 3))
    ry = so3.random_uniform(k2, (2, 3, 3, 3))
    np.testing.assert_allclose(
        so3.velocity(rx, ry, t_1d),
        so3.velocity(rx, ry, t_4d),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        manifolds.geodesic(so3, rx, ry, t_1d),
        manifolds.geodesic(so3, rx, ry, t_4d),
        atol=1e-5,
    )

    # Sphere (B=2, N=3, D=4)
    sphere = manifolds.Sphere()
    sx = sphere.random_uniform(k1, (2, 3, 4))
    sy = sphere.random_uniform(k2, (2, 3, 4))
    np.testing.assert_allclose(
        sphere.velocity(sx, sy, t_1d),
        sphere.velocity(sx, sy, t_3d),
        atol=1e-5,
    )


if __name__ == "__main__":
  absltest.main()
