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

"""Unit tests for fast Gamma, Dirichlet, and Beta random samplers."""

from absl.testing import absltest
from absl.testing import parameterized
from hackable_diffusion.lib import fast_random
import jax
import jax.numpy as jnp
import numpy as np

# Number of samples for statistical tests.
_NUM_SAMPLES = 1_000_000
# Tolerance level. Note that since samplers are biased, tolerance is higher.
_STATISTICAL_TOLERANCE = 1e-2
_RANDOM_SEED = 42


################################################################################
# MARK: Log-Gamma Tests
################################################################################


class LogGammaFastTest(parameterized.TestCase):

  def test_output_shape_from_alpha(self):
    """Shape should be inferred from alpha when shape=None."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.ones((3, 4))
    result = fast_random.log_gamma_fast(key, alpha)
    self.assertEqual(result.shape, (3, 4))

  def test_output_shape_explicit(self):
    """Explicit shape should override alpha shape."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(2.0)
    result = fast_random.log_gamma_fast(key, alpha, shape=(5, 3))
    self.assertEqual(result.shape, (5, 3))

  def test_output_shape_default(self):
    """Default shape should be empty tuple when shape=None."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(2.0)
    result = fast_random.log_gamma_fast(key, alpha)
    self.assertEqual(result.shape, ())

  def test_output_shape_mismatch(self):
    """Default shape should be empty tuple when shape=None."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.ones((7, 32)) * 2.0
    with self.assertRaisesRegex(
        ValueError, "Incompatible types for broadcasting"
    ):
      fast_random.log_gamma_fast(key, alpha, shape=(5, 3))

  def test_output_is_finite(self):
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(2.0)
    result = fast_random.log_gamma_fast(key, alpha, shape=(1000,))
    self.assertTrue(jnp.all(jnp.isfinite(result)))

  @parameterized.parameters(0.5, 1.0, 2.0, 5.0, 10.0)
  def test_mean_matches_expected(self, alpha_val):
    """E[X] = alpha for Gamma(alpha, 1). Check via E[exp(log X)]."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(alpha_val)
    log_samples = fast_random.log_gamma_fast(key, alpha, shape=(_NUM_SAMPLES,))
    samples = jnp.exp(log_samples)
    empirical_mean = jnp.mean(samples)
    np.testing.assert_allclose(
        empirical_mean, alpha_val, rtol=_STATISTICAL_TOLERANCE
    )

  @parameterized.parameters(0.5, 1.0, 2.0, 5.0, 10.0)
  def test_variance_matches_expected(self, alpha_val):
    """Var[X] = alpha for Gamma(alpha, 1)."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(alpha_val)
    log_samples = fast_random.log_gamma_fast(key, alpha, shape=(_NUM_SAMPLES,))
    samples = jnp.exp(log_samples)
    empirical_var = jnp.var(samples)
    np.testing.assert_allclose(
        empirical_var, alpha_val, rtol=_STATISTICAL_TOLERANCE
    )

  def test_different_keys_give_different_samples(self):
    alpha = jnp.array(2.0)
    s1 = fast_random.log_gamma_fast(jax.random.PRNGKey(0), alpha, shape=(100,))
    s2 = fast_random.log_gamma_fast(jax.random.PRNGKey(1), alpha, shape=(100,))
    self.assertFalse(jnp.allclose(s1, s2))


################################################################################
# MARK: Log-Dirichlet / Dirichlet Tests
################################################################################


class DirichletFastTest(parameterized.TestCase):

  def test_log_dirichlet_output_shape(self):
    """log-Dirichlet output shape should be (batch_size, num_components)."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array([1.0, 2.0, 3.0])
    result = fast_random.log_dirichlet_fast(key, alpha, shape=(4,))
    self.assertEqual(result.shape, (4, 3))

  def test_log_dirichlet_default_shape(self):
    """log-Dirichlet output shape should be (num_components,) when shape=None."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array([1.0, 2.0, 3.0])
    result = fast_random.log_dirichlet_fast(key, alpha)
    self.assertEqual(result.shape, (3,))

  def test_dirichlet_sums_to_one(self):
    """Dirichlet samples should sum to 1 along the last axis."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array([2.0, 3.0, 5.0])
    samples = fast_random.dirichlet_fast(key, alpha, shape=(1000,))
    sums = jnp.sum(samples, axis=-1)
    np.testing.assert_allclose(sums, jnp.ones(1000), atol=1e-5)

  def test_log_dirichlet_logsumexp_is_zero(self):
    """log-Dirichlet samples should satisfy logsumexp = 0 (i.e. sum = 1)."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array([1.0, 1.0, 1.0, 1.0])
    log_samples = fast_random.log_dirichlet_fast(key, alpha, shape=(500,))
    logsumexp = jax.nn.logsumexp(log_samples, axis=-1)
    np.testing.assert_allclose(logsumexp, jnp.zeros(500), atol=1e-5)

  def test_dirichlet_all_positive(self):
    """All Dirichlet components should be positive."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array([0.5, 0.5, 0.5])
    samples = fast_random.dirichlet_fast(key, alpha, shape=(1000,))
    self.assertTrue(jnp.all(samples > 0))

  @parameterized.parameters(
      ([1.0, 1.0, 1.0],),
      ([2.0, 3.0, 5.0],),
      ([0.5, 0.5],),
  )
  def test_dirichlet_mean_matches_expected(self, alpha_list):
    """E[X_i] = alpha_i / sum(alpha) for Dirichlet."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(alpha_list)
    samples = fast_random.dirichlet_fast(key, alpha, shape=(_NUM_SAMPLES,))
    empirical_mean = jnp.mean(samples, axis=0)
    expected_mean = alpha / jnp.sum(alpha)
    np.testing.assert_allclose(
        empirical_mean, expected_mean, atol=_STATISTICAL_TOLERANCE
    )

  @parameterized.parameters(
      ([1.0, 1.0, 1.0],),
      ([2.0, 3.0, 5.0],),
      ([0.5, 0.5],),
  )
  def test_dirichlet_variance_matches_expected(self, alpha_list):
    """Var[X_i] = alpha_i * (a0 - alpha_i) / (a0^2 * (a0 + 1))."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha = jnp.array(alpha_list)
    samples = fast_random.dirichlet_fast(key, alpha, shape=(_NUM_SAMPLES,))
    empirical_var = jnp.var(samples, axis=0)
    alpha_0 = jnp.sum(alpha)
    expected_var = alpha * (alpha_0 - alpha) / (alpha_0**2 * (alpha_0 + 1))
    np.testing.assert_allclose(
        empirical_var, expected_var, rtol=_STATISTICAL_TOLERANCE
    )

  def test_dirichlet_consistency_with_log_dirichlet(self):
    """dirichlet_fast should equal exp(log_dirichlet_fast) for same key."""
    key = jax.random.PRNGKey(5)
    alpha = jnp.array([2.0, 3.0, 5.0])
    samples = fast_random.dirichlet_fast(key, alpha, shape=(100,))
    log_samples = fast_random.log_dirichlet_fast(key, alpha, shape=(100,))
    np.testing.assert_allclose(samples, jnp.exp(log_samples), atol=1e-6)


################################################################################
# MARK: Log-Beta Tests
################################################################################


class SampleLogBetaJointTest(parameterized.TestCase):

  def test_output_shape(self):
    """Output shape should be (batch_size, 2) when shape=(batch_size,)."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha, beta = jnp.array(2.0), jnp.array(3.0)
    log_w, log_1mw = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(7, 5)
    )
    self.assertEqual(log_w.shape, (7, 5))
    self.assertEqual(log_1mw.shape, (7, 5))

  def test_log_partition_consistency(self):
    """log(w) and log(1-w) should satisfy exp(log_w) + exp(log_1mw) ≈ 1."""
    key = jax.random.PRNGKey(1)
    alpha, beta = jnp.array(2.0), jnp.array(5.0)
    log_w, log_1mw = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(5000,)
    )
    total = jnp.exp(log_w) + jnp.exp(log_1mw)
    np.testing.assert_allclose(total, jnp.ones(5000), atol=1e-5)

  def test_values_in_valid_range(self):
    """Both log_w and log_1mw should be <= 0 (since w, 1-w ∈ [0, 1])."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha, beta = jnp.array(1.0), jnp.array(1.0)
    log_w, log_1mw = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(5000,)
    )
    self.assertTrue(jnp.all(log_w <= 0.0))
    self.assertTrue(jnp.all(log_1mw <= 0.0))

  @parameterized.parameters(
      (2.0, 5.0),
      (1.0, 1.0),
      (0.5, 0.5),
      (5.0, 2.0),
  )
  def test_mean_matches_expected(self, alpha_val, beta_val):
    """E[W] = alpha / (alpha + beta) for Beta(alpha, beta)."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha, beta = jnp.array(alpha_val), jnp.array(beta_val)
    log_w, _ = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(_NUM_SAMPLES,)
    )
    empirical_mean = jnp.mean(jnp.exp(log_w))
    expected_mean = alpha_val / (alpha_val + beta_val)
    np.testing.assert_allclose(
        empirical_mean, expected_mean, rtol=_STATISTICAL_TOLERANCE
    )

  @parameterized.parameters(
      (2.0, 5.0),
      (1.0, 1.0),
      (0.5, 0.5),
      (5.0, 2.0),
  )
  def test_variance_matches_expected(self, alpha_val, beta_val):
    """Var[W] = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha, beta = jnp.array(alpha_val), jnp.array(beta_val)
    log_w, _ = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(_NUM_SAMPLES,)
    )
    empirical_var = jnp.var(jnp.exp(log_w))
    ab = alpha_val + beta_val
    expected_var = (alpha_val * beta_val) / (ab**2 * (ab + 1))
    np.testing.assert_allclose(
        empirical_var, expected_var, atol=_STATISTICAL_TOLERANCE
    )

  def test_symmetry_of_the_mean(self):
    """Beta(a, b) and Beta(b, a) should have mirrored means."""
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha, beta = jnp.array(2.0), jnp.array(5.0)
    log_w_ab, _ = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(_NUM_SAMPLES,)
    )
    _, log_1mw_ba = fast_random.sample_log_beta_joint(
        key, beta, alpha, shape=(_NUM_SAMPLES,)
    )
    # Beta(a,b) → w has mean a/(a+b). Beta(b,a) → w has mean b/(a+b).
    # So mean(w_ab) ≈ mean(1-w_ba).
    mean_w_ab = jnp.mean(jnp.exp(log_w_ab))
    mean_1mw_ba = jnp.mean(jnp.exp(log_1mw_ba))
    np.testing.assert_allclose(
        mean_w_ab, mean_1mw_ba, rtol=_STATISTICAL_TOLERANCE
    )

  def test_outputs_are_finite(self):
    key = jax.random.PRNGKey(_RANDOM_SEED)
    alpha, beta = jnp.array(0.5), jnp.array(0.5)
    log_w, log_1mw = fast_random.sample_log_beta_joint(
        key, alpha, beta, shape=(5000,)
    )
    self.assertTrue(jnp.all(jnp.isfinite(log_w)))
    self.assertTrue(jnp.all(jnp.isfinite(log_1mw)))

  def test_shape_none_supported(self):
    key = jax.random.PRNGKey(0)
    d = fast_random.log_dirichlet_fast(key, jnp.ones((2, 3)), shape=None)
    self.assertEqual(d.shape, (2, 3))
    b1, b2 = fast_random.sample_log_beta_joint(
        key, jnp.ones((2, 3)), jnp.ones((3,)), shape=None
    )
    self.assertEqual(b1.shape, (2, 3))
    self.assertEqual(b2.shape, (2, 3))
    b1_default, b2_default = fast_random.sample_log_beta_joint(
        key, jnp.ones((2, 3)), jnp.ones((3,))
    )
    self.assertEqual(b1_default.shape, (2, 3))
    self.assertEqual(b2_default.shape, (2, 3))


if __name__ == "__main__":
  absltest.main()
