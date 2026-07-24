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

"""Tests for simplicial corruption processes.

The main difference between categorical and simplicial corruption is that xt is
not of shape [B, T, 1] but [B, T, K]. Instead of tracking a realization of a
certain categorical distribution, we track a realization of a certain Dirichlet
distribution. Therefore, to obtain a token, we must convert the probability
distribution to a token. Here we use the argmax function (we could have also
used categorical sampling).

Apart from this crucial difference, the tests are mostly identical; and
post-corruption projection functions (identity and symmetric) are supported
for simplicial processes, mirroring the discrete case.
"""

import chex
from hackable_diffusion.lib.corruption import simplicial

import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: SimplicialProcessTest
################################################################################


class SimplicialProcessTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.schedule = simplicial.LinearSimplicialSchedule()

    self.num_categories = 5
    self.process = simplicial.SimplicialProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    self.key = jax.random.PRNGKey(42)
    self.batch_size = 256  # large batch size to assert distribution properties
    self.seq_len = 10
    self.shape = (self.batch_size, self.seq_len, 1)
    self.x0 = jax.random.randint(
        self.key,
        shape=self.shape,
        minval=0,
        maxval=self.num_categories,
    )

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_corrupt_different_results_with_scalar_time(self, process_type):
    # At t=0, alpha=1, so xt should be x0.
    if process_type == 'masking':
      process = simplicial.SimplicialProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    else:
      raise ValueError(f'Unknown process type: {process_type}')

    x0 = jnp.ones((2, 2, 1), dtype=jnp.int32) * 2
    time_05 = jnp.ones((1,)) * 0.5
    xt, _ = process.corrupt(self.key, x0, time_05)
    self.assertFalse(jnp.array_equal(xt[0], xt[1]))

  def test_corrupt_different_results_with_batched_time(self):
    # At t=0, alpha=1, so xt should be x0.
    process = simplicial.SimplicialProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    x0 = jnp.ones((2, 2, 10, 1), dtype=jnp.int32) * 2
    time_05 = jnp.ones((2,)) * 0.5
    xt, _ = process.corrupt(self.key, x0, time_05)
    expected_val_0 = jnp.ones((2, 10, 1), dtype=jnp.int32) * xt[0, 0, 0, 0]
    expected_val_1 = jnp.ones((2, 10, 1), dtype=jnp.int32) * xt[1, 0, 0, 0]
    self.assertFalse(jnp.array_equal(xt[0], expected_val_0))
    self.assertFalse(jnp.array_equal(xt[1], expected_val_1))

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_corrupt_at_t0(self, process_type):
    # At t=0, alpha=1, so xt should be x0.
    if process_type == 'masking':
      process = simplicial.SimplicialProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    else:
      raise ValueError(f'Unknown process type: {process_type}')
    time_0 = jnp.zeros((self.batch_size, self.seq_len, 1))
    xt_init, _ = process.corrupt(self.key, self.x0, time_0)
    # Contrary to categorical corruption, simplicial corruption does not
    # preserve x0 at t=0. This is because in that case we have a probability
    # distribution.
    xt_init_amax = jnp.argmax(xt_init, axis=-1)[..., None]
    self.assertTrue(jnp.array_equal(xt_init_amax, self.x0))

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_corrupt_at_t1(self, process_type):
    # At t=1, alpha=0, so xt should be pure noise.
    if process_type == 'masking':
      process = simplicial.SimplicialProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    else:
      raise ValueError(f'Unknown process type: {process_type}')
    time_final = jnp.ones((self.batch_size, self.seq_len, 1))
    key1, _ = jax.random.split(self.key)
    # Use a constant x0 to make it unlikely to be identical to noise.
    x0 = jnp.zeros(self.shape, dtype=jnp.int32)
    xt_final, _ = process.corrupt(key1, x0, time_final)
    xt_final_amax = jnp.argmax(xt_final, axis=-1)[..., None]
    # Check that corruption happened.
    if process_type == 'masking':
      mask_value = self.num_categories
      self.assertTrue(jnp.all(xt_final_amax == jnp.ones_like(x0) * mask_value))
    elif process_type == 'uniform':
      bincount = jnp.bincount(xt_final_amax.flatten())
      bincount_prob = bincount / (self.batch_size * self.seq_len)
      true_prob = 1.0 / self.num_categories
      # Check that the distribution is uniform.
      self.assertTrue(jnp.allclose(bincount_prob, true_prob, atol=0.1))
      # high atol due to limited batch size
    else:
      raise ValueError(f'Unknown process type: {process_type}')

  @parameterized.named_parameters(
      ('masking_high', 'masking', simplicial.PrecisionMode.HIGH),
      ('masking_low', 'masking', simplicial.PrecisionMode.LOW),
      ('uniform_high', 'uniform', simplicial.PrecisionMode.HIGH),
      ('uniform_low', 'uniform', simplicial.PrecisionMode.LOW),
  )
  def test_corrupt_at_t1_with_precision_mode(
      self, process_type, precision_mode
  ):
    # At t=1, alpha=0, so xt should be pure noise.
    if process_type == 'masking':
      process = simplicial.SimplicialProcess(
          schedule=self.schedule,
          invariant_probs=[0.0] * self.num_categories + [1.0],
          num_categories=self.num_categories,
          precision_mode=precision_mode,
      )
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess(
          schedule=self.schedule,
          invariant_probs=[1.0 / self.num_categories] * self.num_categories,
          num_categories=self.num_categories,
          precision_mode=precision_mode,
      )
    else:
      raise ValueError(f'Unknown process type: {process_type}')

    time_final = jnp.ones((self.batch_size, self.seq_len, 1))
    key1, _ = jax.random.split(self.key)
    # Use a constant x0 to make it unlikely to be identical to noise.
    x0 = jnp.zeros(self.shape, dtype=jnp.int32)
    xt_final, _ = process.corrupt(key1, x0, time_final)
    xt_final_amax = jnp.argmax(xt_final, axis=-1)[..., None]
    # Check that corruption happened.
    if process_type == 'masking':
      mask_value = self.num_categories
      self.assertTrue(jnp.all(xt_final_amax == jnp.ones_like(x0) * mask_value))
    elif process_type == 'uniform':
      bincount = jnp.bincount(
          xt_final_amax.flatten(), length=self.num_categories
      )
      bincount_prob = bincount / (self.batch_size * self.seq_len)
      true_prob = 1.0 / self.num_categories
      # Check that the distribution is uniform.
      self.assertTrue(jnp.allclose(bincount_prob, true_prob, atol=0.1))
      # high atol due to limited batch size
    else:
      raise ValueError(f'Unknown process type: {process_type}')

  def test_convert_predictions(self):
    logits = jax.random.normal(self.key, (self.batch_size, self.num_categories))
    prediction = {'logits': logits}
    # xt and time are not used in convert_predictions for CategoricalProcess.
    xt = jnp.empty_like(self.x0)
    time = jnp.empty((self.batch_size, self.seq_len, 1))

    converted = self.process.convert_predictions(prediction, xt, time)
    expected_x0 = jnp.argmax(logits, axis=-1)
    expected_x0 = jnp.expand_dims(expected_x0, axis=-1)

    self.assertIn('x0', converted)
    self.assertIn('logits', converted)
    self.assertTrue(jnp.array_equal(converted['x0'], expected_x0))
    self.assertTrue(jnp.array_equal(converted['logits'], logits))

  def test_convert_predictions_raises_for_wrong_prediction_key(self):
    prediction = {'bad_key': jnp.zeros((self.batch_size, self.num_categories))}
    xt = jnp.empty_like(self.x0)
    time = jnp.empty((self.batch_size, 1))
    with self.assertRaises(KeyError):
      self.process.convert_predictions(prediction, xt, time)

  def test_non_valid_unused_token_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'unused_token must be outside of the range of the vocabulary.',
    ):
      simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule,
          num_categories=self.num_categories,
          unused_token=0,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='uniform',
          factory_name='uniform_process',
          num_categories=5,
          expected_process_num_categories=5,
          expected_probs=[0.2] * 5,
      ),
      dict(
          testcase_name='masking',
          factory_name='masking_process',
          num_categories=5,
          expected_process_num_categories=6,
          expected_probs=[0.0] * 5 + [1.0],
      ),
  )
  def test_factory_methods(
      self,
      factory_name,
      num_categories,
      expected_process_num_categories,
      expected_probs,
  ):
    expected_probs = jnp.array(expected_probs)
    factory = getattr(simplicial.SimplicialProcess, factory_name)
    process = factory(schedule=self.schedule, num_categories=num_categories)
    self.assertEqual(
        process.process_num_categories, expected_process_num_categories
    )
    self.assertEqual(process.num_categories, num_categories)
    self.assertTrue(jnp.allclose(process.invariant_probs_vec, expected_probs))

  def test_factory_methods_raise_for_invalid_num_categories(self):
    with self.assertRaisesRegex(ValueError, 'num_categories must be positive'):
      simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule, num_categories=0
      )
    with self.assertRaisesRegex(ValueError, 'num_categories must be positive'):
      simplicial.SimplicialProcess.masking_process(
          schedule=self.schedule, num_categories=0
      )

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_is_masking(self, process_type):
    if process_type == 'masking':
      process = simplicial.SimplicialProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
      self.assertTrue(process.is_masking)
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
      self.assertFalse(process.is_masking)
    else:
      raise ValueError(f'Unknown process type: {process_type}')

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_is_masking_by_hand_crafted_invariant_probs(self, process_type):
    if process_type == 'masking':
      process = simplicial.SimplicialProcess(
          schedule=self.schedule,
          invariant_probs=[0.0] * self.num_categories + [1.0],  # masking
          num_categories=self.num_categories,
      )
      self.assertTrue(process.is_masking)
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess(
          schedule=self.schedule,
          invariant_probs=[1.0 / self.num_categories] * self.num_categories,
          num_categories=self.num_categories,
      )
      self.assertFalse(process.is_masking)
    else:
      raise ValueError(f'Unknown process type: {process_type}')

  # ---------------------------------------------------------------------------
  # MARK: h-function tests
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      dict(testcase_name='t_half_linear', time=0.5, expected_h=1.0),
      dict(
          testcase_name='t_three_quarters_linear',
          time=0.75,
          expected_h=1.0 / 3.0,
      ),
      dict(testcase_name='t_quarter_linear', time=0.25, expected_h=3.0),
  )
  def test_h_function_values(self, time, expected_h):
    """h(t) = alpha(t) / (1-alpha(t)) for linear schedule alpha(t)=1-t.

    For the linear schedule alpha(t) = 1-t:
      h(t) = (1-t) / t
    so h(0.25)=3, h(0.5)=1, h(0.75)=1/3.

    Args:
      time: The time of the corruption.
      expected_h: The expected value of the h-function.

    Returns:
      None
    """
    # h() requires at least a 1D array (ktyping constraint).
    t = jnp.array([time])
    h_val = self.process.h(t)
    self.assertAlmostEqual(float(h_val[0]), expected_h, places=4)

  def test_h_function_is_decreasing(self):
    """h(t) must be a strictly decreasing function of t."""
    times = jnp.linspace(0.01, 0.99, 20)
    # h() expects a batch dimension; pass the whole array at once.
    h_vals = self.process.h(times)
    # Each step should be smaller than the previous.
    diffs = jnp.diff(h_vals)
    self.assertTrue(jnp.all(diffs < 0))

  # ---------------------------------------------------------------------------
  # MARK: sample_from_invariant tests
  # ---------------------------------------------------------------------------

  def test_sample_from_invariant_shape_and_normalization(self):
    """sample_from_invariant should return valid log-probabilities."""
    data_spec = jnp.zeros((self.batch_size, self.seq_len, 1))
    log_samples = self.process.sample_from_invariant(
        self.key, data_spec=data_spec
    )
    # Shape should be [B, T, K].
    expected_shape = (self.batch_size, self.seq_len, self.num_categories)
    self.assertEqual(log_samples.shape, expected_shape)
    # Must be valid log-probabilities: logsumexp == 0.
    log_sum = jax.nn.logsumexp(log_samples, axis=-1)
    self.assertTrue(jnp.allclose(log_sum, jnp.zeros_like(log_sum), atol=1e-5))

  def test_sample_from_invariant_masking_concentrates_on_mask(self):
    """For masking process at t=1, invariant samples concentrate on mask token."""
    process = simplicial.SimplicialProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    data_spec = jnp.zeros((self.batch_size, self.seq_len, 1))
    log_samples = process.sample_from_invariant(self.key, data_spec=data_spec)
    # All mass should be on the last (mask) token.
    argmax = jnp.argmax(log_samples, axis=-1)
    self.assertTrue(jnp.all(argmax == self.num_categories))

  # ---------------------------------------------------------------------------
  # MARK: Unused-token passthrough tests
  # (discrete_test has test_unused_mask_gives_always_false_on_other_masks)
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_unused_token_is_preserved_at_all_times(self, process_type):
    """Positions with unused_token in x0 must be passed through unchanged."""
    process = {
        'masking': simplicial.SimplicialProcess.masking_process(
            schedule=self.schedule, num_categories=self.num_categories
        ),
        'uniform': simplicial.SimplicialProcess.uniform_process(
            schedule=self.schedule, num_categories=self.num_categories
        ),
    }[process_type]

    # Build a sequence [1, UNUSED, 2, UNUSED, 3, 4] (shape [1, 6, 1]).
    x0 = jnp.array(
        [1, process.unused_token, 2, process.unused_token, 3, 4],
        dtype=jnp.int32,
    ).reshape((1, 6, 1))
    # xt has shape [1, 6, K]; unused positions are filled with unused_token.
    unused_positions = [False, True, False, True, False, False]

    for t_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
      time = jnp.ones((1,)) * t_val
      xt, _ = process.corrupt(self.key, x0, time)
      # Shape is [1, 6, K]; check each position individually.
      for i, is_unused in enumerate(unused_positions):
        slot = xt[0, i, :]  # shape [K]
        if is_unused:
          # The entire slot must be filled with the unused_token.
          self.assertTrue(
              jnp.all(slot == process.unused_token),
              msg=f'Unused token not preserved at t={t_val}, position {i}',
          )
        else:
          # Non-unused positions must *not* be the unused_token.
          self.assertFalse(
              jnp.all(slot == process.unused_token),
              msg=(
                  f'Valid position wrongly set to unused_token at t={t_val},'
                  f' pos {i}'
              ),
          )
# ---------------------------------------------------------------------------
  # MARK: Schedule evaluation tests
  # ---------------------------------------------------------------------------

  def test_schedule_evaluate_returns_alpha(self):
    """Schedule evaluation should include alpha."""
    time = jnp.array([0.5])
    info = self.process.schedule.evaluate(time)
    self.assertIn('alpha', info)
    self.assertIn('time', info)
    # For linear schedule alpha(t) = 1 - t.
    expected_alpha = 1.0 - time
    chex.assert_trees_all_close(info['alpha'], expected_alpha, atol=1e-6)


################################################################################
# MARK: SimplicialPostCorruptionTest
################################################################################


class SimplicialPostCorruptionTest(parameterized.TestCase):
  """Tests for SimplicialPostCorruptionFn classes."""

  def _make_log_x(self, batch=2, n=4, k=3):
    """Helper: a random log-prob array of shape (batch, n, n, k)."""
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, shape=(batch, n, n, k))
    return logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)

  # ---------------------------------------------------------------------------
  # IdentitySimplicialPostCorruptionFn
  # ---------------------------------------------------------------------------

  def test_identity_fn_is_identity(self):
    """IdentitySimplicialPostCorruptionFn must be a no-op."""
    fn = simplicial.IdentitySimplicialPostCorruptionFn()
    log_x = self._make_log_x()
    log_y = fn(log_x)
    chex.assert_trees_all_close(log_y, log_x)

  # ---------------------------------------------------------------------------
  # SymmetricSimplicialPostCorruptionFn
  # ---------------------------------------------------------------------------

  def test_symmetric_fn_produces_symmetric_output(self):
    """After projection, log_y[b, i, j, :] == log_y[b, j, i, :] for all i != j."""
    fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    log_x = self._make_log_x(batch=2, n=5, k=4)
    log_y = fn(log_x)
    log_y_transpose = jnp.swapaxes(log_y, 1, 2)
    chex.assert_trees_all_close(log_y, log_y_transpose, atol=1e-5)

  def test_symmetric_fn_is_log_normalized(self):
    """Every off-diagonal row of the output must be a valid log-prob vector.

    This holds trivially because the output is a direct copy of an upper-
    triangle log-prob vector (already normalised).  The test exists to guard
    against future regressions.
    """
    fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    batch_size, n, k = 2, 4, 3
    log_x = self._make_log_x(batch=batch_size, n=n, k=k)
    log_y = fn(log_x)
    # Check off-diagonal positions only; diagonal is set to uniform separately.
    off_diag = ~jnp.eye(n, dtype=jnp.bool_)
    for b in range(batch_size):
      for i in range(n):
        for j in range(n):
          if not off_diag[i, j]:
            continue
          log_sum = jax.nn.logsumexp(log_y[b, i, j, :])
          chex.assert_trees_all_close(log_sum, jnp.zeros(()), atol=1e-5)

  def test_symmetric_fn_copies_upper_triangle_exactly(self):
    """Upper-triangle values must be preserved; lower triangle must match them.

    Specifically, log_y[b, i, j] must equal log_x[b, i, j] for i < j, and
    log_y[b, j, i] must equal log_x[b, i, j] for j > i.  No blending should
    occur (contrast with a geometric-mean approach).
    """
    fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    batch, n, k = 2, 4, 3
    log_x = self._make_log_x(batch=batch, n=n, k=k)
    log_y = fn(log_x)
    for b in range(batch):
      for i in range(n):
        for j in range(i + 1, n):  # upper triangle only
          # Upper-triangle entry is unchanged.
          chex.assert_trees_all_close(
              log_y[b, i, j, :], log_x[b, i, j, :], atol=1e-6
          )
          # Lower-triangle entry is copied from upper.
          chex.assert_trees_all_close(
              log_y[b, j, i, :], log_x[b, i, j, :], atol=1e-6
          )

  def test_symmetric_fn_zeroes_diagonal(self):
    """Diagonal entries should be set to the no-edge vector (category 0)."""
    fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    batch, n, k = 2, 4, 3
    log_x = self._make_log_x(batch=batch, n=n, k=k)
    log_y = fn(log_x)
    # No-edge vector: all mass on category 0, i.e. [0, -inf, -inf, ...]
    expected = jnp.full((k,), -1e9, dtype=log_y.dtype).at[0].set(0.0)
    for b in range(batch):
      for i in range(n):
        chex.assert_trees_all_close(
            log_y[b, i, i, :],
            expected,
            atol=1e-5,
        )

  def test_symmetric_fn_raises_on_wrong_dims(self):
    """Should raise ValueError if input is not 4-D or not square."""
    fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    with self.assertRaisesRegex(ValueError, 'Expected 4D'):
      fn(jnp.zeros((2, 3, 4)))  # 3-D input
    with self.assertRaisesRegex(ValueError, 'Spatial dimensions must be equal'):
      fn(jnp.zeros((2, 3, 4, 5)))  # non-square

  # ---------------------------------------------------------------------------
  # SimplicialProcess with post_corruption_fn
  # ---------------------------------------------------------------------------

  def test_corrupt_with_symmetric_fn_produces_symmetric_output(self):
    """SimplicialProcess.corrupt() must respect post_corruption_fn.

    When a SymmetricSimplicialPostCorruptionFn is plugged into the process,
    the corrupted xt must satisfy xt[b, i, j, :] == xt[b, j, i, :] for i != j.

    We use a (batch=1, N=3, N=3) adjacency layout.
    """
    schedule = simplicial.LinearSimplicialSchedule()

    fn = simplicial.SymmetricSimplicialPostCorruptionFn()
    process = simplicial.SimplicialProcess.uniform_process(
        schedule=schedule, num_categories=4, post_corruption_fn=fn
    )
    # x0: integer tokens for a 3-node graph, shape (1, 3, 3, 1)
    x0 = jax.random.randint(
        jax.random.PRNGKey(7), shape=(1, 3, 3, 1), minval=0, maxval=4
    )
    time = jnp.array([0.3])
    xt, _ = process.corrupt(jax.random.PRNGKey(1), x0, time)
    # xt has shape (1, 3, 3, 4) -- log-prob array.
    xt_transpose = jnp.swapaxes(xt, 1, 2)
    n = xt.shape[1]
    off_diag_mask = ~jnp.eye(n, dtype=jnp.bool_)[None, :, :, None]
    chex.assert_trees_all_close(
        jnp.where(off_diag_mask, xt, 0.0),
        jnp.where(off_diag_mask, xt_transpose, 0.0),
        atol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
