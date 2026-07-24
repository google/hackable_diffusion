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

"""Tests for simplicial step sampler."""

import chex
from hackable_diffusion.lib import fast_random
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib.corruption import simplicial
from hackable_diffusion.lib.sampling import simplicial_step_sampler
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = hd_api.DiffusionStep
StepInfo = hd_api.StepInfo
SimplicialProcess = simplicial.SimplicialProcess
SimplicialDDIMStep = simplicial_step_sampler.SimplicialDDIMStep
log_beta_shrinkage = simplicial_step_sampler.log_beta_shrinkage

################################################################################
# MARK: Tests
################################################################################


class BetaShrinkageTest(parameterized.TestCase):
  """Tests for beta shrinkage."""

  def test_identity(self):
    key = jax.random.PRNGKey(0)
    log_x = jax.nn.log_softmax(jax.random.normal(key, (2, 4, 3)))
    concentration = jnp.array(10.0)
    log_y = log_beta_shrinkage(key, log_x, concentration, kappa=1.0)
    chex.assert_trees_all_close(log_x, log_y)

  def test_shape(self):
    key = jax.random.PRNGKey(0)
    log_x = jax.nn.log_softmax(jax.random.normal(key, (2, 4, 3)))
    concentration = jnp.array(10.0)
    log_y = log_beta_shrinkage(key, log_x, concentration, kappa=0.5)
    self.assertEqual(log_x.shape, log_y.shape)
    # Check that it's still normalized (logsumexp == 0)
    chex.assert_trees_all_close(
        jax.nn.logsumexp(log_y, axis=-1), jnp.zeros(log_x.shape[:-1]), atol=1e-5
    )

  def test_stochasticity(self):
    key = jax.random.PRNGKey(0)
    log_x = jax.nn.log_softmax(jax.random.normal(key, (2, 4, 3)))
    concentration = jnp.array(10.0)
    key1, key2 = jax.random.split(key)
    log_y1 = log_beta_shrinkage(key1, log_x, concentration, kappa=0.5)
    log_y2 = log_beta_shrinkage(key2, log_x, concentration, kappa=0.5)
    # They should be different
    self.assertGreater(jnp.abs(log_y1 - log_y2).sum(), 0.0)

  def test_distribution(self):
    key = jax.random.PRNGKey(0)
    # Target concentration
    alpha = jnp.array([1.0, 2.0, 3.0])
    kappa = 0.5
    target_alpha = alpha * kappa  # [0.5, 1.0, 1.5]

    # Generate many samples to check distribution
    n_samples = 20_000

    key, subkey = jax.random.split(key)
    log_x = fast_random.log_dirichlet_fast(subkey, alpha, shape=(n_samples,))

    _, subkey = jax.random.split(key)
    log_y = log_beta_shrinkage(subkey, log_x, alpha, kappa=kappa)
    y = jnp.exp(log_y)

    # Theoretical moments for Dir(target_alpha)
    sum_alpha = jnp.sum(target_alpha)
    expected_mean = target_alpha / sum_alpha
    expected_var = (
        target_alpha
        * (sum_alpha - target_alpha)
        / (sum_alpha**2 * (sum_alpha + 1))
    )

    # Empirical moments
    empirical_mean = jnp.mean(y, axis=0)
    empirical_var = jnp.var(y, axis=0)

    # Check closeness
    chex.assert_trees_all_close(empirical_mean, expected_mean, atol=5e-3)
    chex.assert_trees_all_close(empirical_var, expected_var, atol=1e-3)


class SimplicialStepSamplerTest(parameterized.TestCase):
  """Tests for simplicial step samplers."""

  def setUp(self):
    super().setUp()
    self.schedule = simplicial.LinearSimplicialSchedule()
    self.num_categories = 4
    self.process = SimplicialProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.normal(key, (2, 4, self.num_categories))
    self.ddim_step = SimplicialDDIMStep(
        corruption_process=self.process, churn=0.0
    )
    self.churn_step = SimplicialDDIMStep(
        corruption_process=self.process, churn=0.5
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 1.
    logits = jnp.zeros(xt.shape[:-1] + (self.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    for sampler in [self.churn_step, self.ddim_step]:
      initial_step_info = StepInfo(
          step=0,
          time=jnp.array([1.0, 1.0])[:, None, None],
          rng=jax.random.PRNGKey(0),
      )
      initial_step = sampler.initialize(
          initial_noise=self.initial_noise,
          initial_step_info=initial_step_info,
      )

      chex.assert_trees_all_equal(
          initial_step,
          DiffusionStep(
              xt=self.initial_noise,
              step_info=initial_step_info,
              aux={'logits': self.initial_noise},
          ),
      )

  def test_update_shapes(self):
    for sampler in [self.churn_step, self.ddim_step]:
      initial_step_info = StepInfo(
          step=0,
          time=jnp.array([0.5, 0.5])[:, None, None],
          rng=jax.random.PRNGKey(0),
      )
      initial_step = sampler.initialize(
          initial_noise=self.initial_noise,
          initial_step_info=initial_step_info,
      )
      prediction = self._dummy_inference_fn(
          xt=initial_step.xt,
          conditioning={},
          time=initial_step.step_info.time,
      )

      next_step_info = StepInfo(
          step=1,
          time=jnp.array([0.1, 0.1])[:, None, None],
          rng=jax.random.PRNGKey(1),
      )
      next_step = sampler.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

      self.assertEqual(next_step.xt.shape, self.initial_noise.shape)
      self.assertEqual(next_step.xt.dtype, self.initial_noise.dtype)

  def test_finalize(self):
    for sampler in [self.churn_step, self.ddim_step]:
      initial_step_info = StepInfo(
          step=0,
          time=jnp.array([0.1, 0.1])[:, None, None],
          rng=jax.random.PRNGKey(0),
      )
      initial_step = sampler.initialize(
          initial_noise=self.initial_noise,
          initial_step_info=initial_step_info,
      )
      prediction = self._dummy_inference_fn(
          xt=initial_step.xt,
          conditioning={},
          time=initial_step.step_info.time,
      )

      last_step_info = StepInfo(
          step=1,
          time=jnp.array([0.0, 0.0])[:, None, None],
          rng=jax.random.PRNGKey(1),
      )
      final_step = sampler.finalize(
          prediction=prediction,
          current_step=initial_step,
          last_step_info=last_step_info,
      )

      self.assertEqual(final_step.xt.shape, self.initial_noise.shape)

  def test_churn_variation(self):
    # Test that different churn values yield different results
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )

    next_step_ddim = self.ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )
    next_step_churn = self.churn_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    # They should have the same shape but different values
    self.assertEqual(next_step_ddim.xt.shape, next_step_churn.xt.shape)
    # With high probability they are different
    diff = jnp.abs(next_step_ddim.xt - next_step_churn.xt).sum()
    self.assertGreater(diff, 0.0)

  @parameterized.named_parameters(
      ('ddim', 'ddim_step'),
      ('churn', 'churn_step'),
  )
  def test_update_output_is_log_normalized(self, sampler_attr):
    """The updated xt should remain a valid log-prob simplex (logsumexp == 0)."""
    sampler = getattr(self, sampler_attr)
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.8, 0.8])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    # Start from a proper log-prob distribution.
    log_noise = jax.nn.log_softmax(
        jax.random.normal(jax.random.PRNGKey(7), self.initial_noise.shape)
    )
    initial_step = sampler.initialize(
        initial_noise=log_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.3, 0.3])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step = sampler.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )
    log_sum = jax.nn.logsumexp(next_step.xt, axis=-1)
    chex.assert_trees_all_close(log_sum, jnp.zeros_like(log_sum), atol=1e-4)

  def test_update_aux_contains_logits(self):
    """The update step should store predicted logits in aux['logits']."""
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step = self.ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )
    self.assertIn('logits', next_step.aux)
    expected_logits_shape = self.initial_noise.shape[:-1] + (
        self.num_categories,
    )
    self.assertEqual(next_step.aux['logits'].shape, expected_logits_shape)

  def test_convergence_to_predicted_class_at_t0(self):
    """With a fully confident model, the predicted logits peak on category 1.

    Note: xt itself mixes logits with Beta-sampled weights so it won't be
    exactly one-hot even at small t.  We therefore check aux['logits'], which
    is the direct model output stored by update().
    """
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    log_noise = jax.nn.log_softmax(
        jax.random.normal(jax.random.PRNGKey(5), self.initial_noise.shape)
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=log_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )
    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step = self.ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )
    # aux['logits'] is the direct model output — should always peak on 1.
    argmax = jnp.argmax(next_step.aux['logits'], axis=-1)
    self.assertTrue(jnp.all(argmax == 1))

  def test_full_churn_step_is_stochastic(self):
    """Verify that full churn is stochastic and produces valid outputs.

    churn=1.0 (full stochasticity, κ=0) produces different results with
    different keys, and the output remains a valid log-prob simplex.

    This tests the safety_epsilon fix: without it, churn=1.0 causes
    Beta(0, b) which is degenerate in JAX and produces NaN.
    """
    full_churn_step = SimplicialDDIMStep(
        corruption_process=self.process, churn=1.0
    )
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    log_noise = jax.nn.log_softmax(
        jax.random.normal(jax.random.PRNGKey(3), self.initial_noise.shape)
    )
    initial_step = full_churn_step.initialize(
        initial_noise=log_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Two calls with different keys produce different outputs.
    step_a = full_churn_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1, 0.1])[:, None, None],
            rng=jax.random.PRNGKey(10),
        ),
    )
    step_b = full_churn_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1, 0.1])[:, None, None],
            rng=jax.random.PRNGKey(99),
        ),
    )
    diff = jnp.abs(step_a.xt - step_b.xt).sum()
    self.assertGreater(diff, 0.0)
    # Output is still properly normalized.
    log_sum = jax.nn.logsumexp(step_a.xt, axis=-1)
    chex.assert_trees_all_close(log_sum, jnp.zeros_like(log_sum), atol=1e-4)

  # ---------------------------------------------------------------------------
  # MARK: post_corruption_fn tests
  # ---------------------------------------------------------------------------

  def test_update_applies_post_corruption_fn(self):
    """SimplicialDDIMStep.update() must call post_corruption_fn on new_xt.

    We use a SymmetricSimplicialPostCorruptionFn and an adjacency-style
    (batch=1, N=3, N=3, K) input.  After each update() the output must satisfy
    xt[b, i, j, :] == xt[b, j, i, :] for all off-diagonal (i, j).
    """
    schedule = simplicial.LinearSimplicialSchedule()
    post_fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    process = simplicial.SimplicialProcess.uniform_process(
        schedule=schedule,
        num_categories=self.num_categories,
        post_corruption_fn=post_fn,
    )
    sampler = SimplicialDDIMStep(corruption_process=process, churn=0.5)

    # Build a (1, 3, 3, K) log-noise by symmetrising random logits.
    n = 3
    k = process.process_num_categories
    raw_logits = jax.random.normal(jax.random.PRNGKey(5), shape=(1, n, n, k))
    log_noise = raw_logits - jax.nn.logsumexp(
        raw_logits, axis=-1, keepdims=True
    )
    # Symmetrise the initial noise so initialize() is consistent.
    log_noise = post_fn(log_noise)

    initial_step = sampler.initialize(
        initial_noise=log_noise,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.5]),
            rng=jax.random.PRNGKey(0),
        ),
    )
    # Dummy predictor: always predict uniform logits.
    prediction = {'logits': jnp.zeros_like(log_noise)}
    step = sampler.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.2]),
            rng=jax.random.PRNGKey(1),
        ),
    )

    xt = step.xt  # (1, 3, 3, K)
    xt_transpose = jnp.swapaxes(xt, 1, 2)
    off_diag_mask = ~jnp.eye(n, dtype=jnp.bool_)[None, :, :, None]
    chex.assert_trees_all_close(
        jnp.where(off_diag_mask, xt, 0.0),
        jnp.where(off_diag_mask, xt_transpose, 0.0),
        atol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
