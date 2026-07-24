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

"""Tests for diffusion_early_stopping."""

import dataclasses

import chex
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib.sampling import diffusion_early_stopping
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Helpers
################################################################################


def _make_step_info():
  """Creates a dummy StepInfo for testing."""
  return hd_api.StepInfo(
      step=jnp.int32(0),
      time=jnp.float32(0.5),
      rng=jax.random.PRNGKey(0),
  )


def _make_diffusion_step(
    xt: jax.Array,
    aux: dict | None = None,
) -> hd_api.DiffusionStep:
  """Creates a DiffusionStep with a dummy StepInfo."""
  return hd_api.DiffusionStep(
      xt=xt,
      step_info=_make_step_info(),
      aux=aux if aux is not None else {},
  )


################################################################################
# MARK: Tests
################################################################################


class DiffusionNoEarlyStopFnTest(parameterized.TestCase):
  """Tests for DiffusionNoEarlyStopFn."""

  def test_never_stops(self):
    fn = diffusion_early_stopping.DiffusionNoEarlyStopFn()
    step = _make_diffusion_step(xt=jnp.ones((4, 8)))
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (4,))
    self.assertTrue(jnp.all(~result))

  @parameterized.named_parameters(
      ('batch_1', 1),
      ('batch_8', 8),
      ('batch_32', 32),
  )
  def test_returns_correct_shape(self, batch_size: int):
    fn = diffusion_early_stopping.DiffusionNoEarlyStopFn()
    step = _make_diffusion_step(xt=jnp.ones((batch_size, 4)))
    result = fn.should_stop(
        step=jnp.int32(5),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (batch_size,))
    self.assertEqual(result.dtype, jnp.bool_)


class DiffusionEntropyEarlyStopFnTest(parameterized.TestCase):
  """Tests for DiffusionEntropyEarlyStopFn."""

  def _make_step_with_logits(
      self,
      logits: jax.Array,
      logits_key: str = 'logits',
  ) -> hd_api.DiffusionStep:
    """Helper: creates a DiffusionStep with logits in aux."""
    batch_size, seq_len = logits.shape[:2]
    return _make_diffusion_step(
        xt=jnp.ones((batch_size, seq_len, 1)),
        aux={logits_key: logits},
    )

  def test_low_entropy_stops(self):
    """When one class dominates, entropy is ~0 and should stop."""
    # Logits that produce a nearly one-hot distribution → near-zero entropy.
    # Shape: [B=2, L=3, V=4]
    logits = jnp.array([
        [[100.0, -100.0, -100.0, -100.0]] * 3,
        [[100.0, -100.0, -100.0, -100.0]] * 3,
    ])
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.05,
    )
    step = self._make_step_with_logits(logits)
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (2,))
    # Both batch elements should stop (entropy ≈ 0 < 0.05).
    self.assertTrue(jnp.all(result))

  def test_high_entropy_continues(self):
    """When the distribution is uniform, entropy is high → should not stop."""
    # Uniform logits → max entropy for V=4 is ln(4) ≈ 1.386.
    logits = jnp.zeros((2, 3, 4))
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.05,
    )
    step = self._make_step_with_logits(logits)
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (2,))
    # Both batch elements should NOT stop.
    self.assertTrue(jnp.all(~result))

  def test_at_threshold_stops(self):
    """Entropy exactly at the threshold should stop (<=)."""
    # Build logits that produce a specific entropy.  We use a hand-crafted
    # probability vector and compute logits = log(probs).
    # For a 2-class distribution with p = [0.99, 0.01]:
    # H ≈ -0.99*ln(0.99) - 0.01*ln(0.01) ≈ 0.056
    # We'll set the threshold to match so that stopping occurs.
    probs = jnp.array([0.99, 0.01])
    entropy_value = float(-jnp.sum(probs * jnp.log(probs)))
    logits = jnp.log(probs)
    # Tile to [B=1, L=1, V=2]
    logits = logits[None, None, :]
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=entropy_value,
    )
    step = self._make_step_with_logits(logits)
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (1,))
    # Exactly at threshold → should stop.
    self.assertTrue(result[0])

  def test_per_batch_mixed_stopping(self):
    """Some batch elements stop while others continue."""
    # Batch element 0: near one-hot → entropy ≈ 0 → should stop.
    # Batch element 1: uniform → entropy = ln(4) ≈ 1.386 → should not stop.
    logits = jnp.array([
        [[100.0, -100.0, -100.0, -100.0]] * 4,  # low entropy
        [[0.0, 0.0, 0.0, 0.0]] * 4,  # high entropy
    ])
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.1,
    )
    step = self._make_step_with_logits(logits)
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(result[0])  # low entropy → stopped
    self.assertFalse(result[1])  # high entropy → continue

  def test_custom_logits_key(self):
    """The logits_key attribute should control which aux key is read."""
    logits = jnp.array([
        [[100.0, -100.0, -100.0, -100.0]] * 3,
    ])
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.05,
        logits_key='my_logits',
    )
    step = self._make_step_with_logits(logits, logits_key='my_logits')
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    self.assertTrue(jnp.all(result))

  def test_default_threshold(self):
    """The default entropy_threshold is 0.05."""
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn()
    self.assertEqual(fn.entropy_threshold, 0.05)
    self.assertEqual(fn.logits_key, 'logits')

  def test_uniform_entropy_value(self):
    """Cross-check: uniform logits over V classes → entropy = ln(V)."""
    vocab_size = 8
    logits = jnp.zeros((1, 5, vocab_size))
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=float(jnp.log(vocab_size) + 0.01),
    )
    step = self._make_step_with_logits(logits)
    # Entropy = ln(V) which equals the threshold exactly,
    # so <= should stop.
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    chex.assert_shape(result, (1,))
    self.assertTrue(result[0])

  def test_ignores_step_and_previous(self):
    """Entropy early stop should not depend on step or previous_step."""
    logits = jnp.zeros((1, 3, 4))
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.05,
    )
    step = self._make_step_with_logits(logits)
    # Different step indices and previous_step should produce same result.
    result_a = fn.should_stop(
        step=jnp.int32(0),
        current_step=step,
        previous_step=step,
    )
    different_prev = self._make_step_with_logits(jnp.ones_like(logits) * 100.0)
    result_b = fn.should_stop(
        step=jnp.int32(999),
        current_step=step,
        previous_step=different_prev,
    )
    chex.assert_trees_all_equal(result_a, result_b)

  def test_raises_value_error_for_invalid_xt_shape(self):
    """Should raise ValueError if xt length of shape is not 3."""
    fn = diffusion_early_stopping.DiffusionEntropyEarlyStopFn()
    step = _make_diffusion_step(
        xt=jnp.ones((2, 4)), aux={'logits': jnp.zeros((2, 4, 3))}
    )
    with self.assertRaisesRegex(
        ValueError, r'xt must have shape \(batch_size, seq_len, 1\) but got'
    ):
      fn.should_stop(
          step=jnp.int32(0),
          current_step=step,
          previous_step=step,
      )


class DiffusionTokenStabilityEarlyStopFnTest(parameterized.TestCase):
  """Tests for DiffusionTokenStabilityEarlyStopFn."""

  def setUp(self):
    super().setUp()
    prev_tokens = jnp.array([[0, 1, 2], [3, 2, 1]])
    self.prev_tokens = jnp.reshape(prev_tokens, (2, 3, 1))

  def test_stable_tokens_stop(self):
    """When argmax(logits) matches previous_step.xt, should stop."""
    logits = jnp.full((2, 3, 4), -100.0)
    for b in range(2):
      for l in range(3):
        logits = logits.at[b, l, self.prev_tokens[b, l]].set(100.0)

    previous_step = _make_diffusion_step(xt=self.prev_tokens)
    current_step = _make_diffusion_step(
        xt=self.prev_tokens, aux={'logits': logits}
    )

    fn = diffusion_early_stopping.DiffusionTokenStabilityEarlyStopFn()
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=current_step,
        previous_step=previous_step,
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(jnp.all(result))

  def test_unstable_tokens_continue(self):
    """When argmax(logits) differs from previous_step.xt, should not stop."""
    logits = jnp.full((2, 3, 4), -100.0).at[:, :, 3].set(100.0)

    previous_step = _make_diffusion_step(xt=self.prev_tokens)
    current_step = _make_diffusion_step(
        xt=self.prev_tokens, aux={'logits': logits}
    )

    fn = diffusion_early_stopping.DiffusionTokenStabilityEarlyStopFn()
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=current_step,
        previous_step=previous_step,
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(jnp.all(~result))

  def test_per_batch_mixed_stability(self):
    """Batch element 0 is stable, batch element 1 is unstable."""
    logits = jnp.full((2, 3, 4), -100.0)
    for l in range(3):
      logits = logits.at[0, l, self.prev_tokens[0, l]].set(100.0)
    logits = logits.at[1, :, 0].set(100.0)

    previous_step = _make_diffusion_step(xt=self.prev_tokens)
    current_step = _make_diffusion_step(
        xt=self.prev_tokens, aux={'logits': logits}
    )

    fn = diffusion_early_stopping.DiffusionTokenStabilityEarlyStopFn()
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=current_step,
        previous_step=previous_step,
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(result[0])
    self.assertFalse(result[1])

  def test_custom_logits_key(self):
    """Should read logits from custom logits_key in aux."""
    logits = jnp.full((2, 3, 4), -100.0)
    for b in range(2):
      for l in range(3):
        logits = logits.at[b, l, self.prev_tokens[b, l]].set(100.0)

    previous_step = _make_diffusion_step(xt=self.prev_tokens)
    current_step = _make_diffusion_step(
        xt=self.prev_tokens, aux={'my_logits': logits}
    )

    fn = diffusion_early_stopping.DiffusionTokenStabilityEarlyStopFn(
        logits_key='my_logits'
    )
    result = fn.should_stop(
        step=jnp.int32(0),
        current_step=current_step,
        previous_step=previous_step,
    )
    self.assertTrue(jnp.all(result))

  def test_raises_value_error_for_invalid_prev_tokens_shape(self):
    """Should raise ValueError if prev_tokens shape is not (batch_size, seq_len, 1)."""
    fn = diffusion_early_stopping.DiffusionTokenStabilityEarlyStopFn()

    # Case 1: 2D prev_tokens shape (2, 4)
    prev_step_2d = _make_diffusion_step(xt=jnp.ones((2, 4)))
    current_step = _make_diffusion_step(
        xt=jnp.ones((2, 4, 1)), aux={'logits': jnp.zeros((2, 4, 3))}
    )
    with self.assertRaisesRegex(
        ValueError,
        r'prev_tokens must have shape \(batch_size, seq_len, 1\) but got',
    ):
      fn.should_stop(
          step=jnp.int32(0),
          current_step=current_step,
          previous_step=prev_step_2d,
      )

    # Case 2: 3D prev_tokens shape with trailing dim != 1 (2, 4, 2)
    prev_step_bad_dim = _make_diffusion_step(xt=jnp.ones((2, 4, 2)))
    with self.assertRaisesRegex(
        ValueError,
        r'prev_tokens must have shape \(batch_size, seq_len, 1\) but got',
    ):
      fn.should_stop(
          step=jnp.int32(0),
          current_step=current_step,
          previous_step=prev_step_bad_dim,
      )


class DiffusionChainedEarlyStopFnTest(parameterized.TestCase):
  """Tests for DiffusionChainedEarlyStopFn."""

  def test_raises_value_error_if_empty(self):
    """Should raise ValueError if instantiated with an empty list."""
    with self.assertRaisesRegex(
        ValueError, 'requires at least one EarlyStopFn'
    ):
      diffusion_early_stopping.DiffusionChainedEarlyStopFn(early_stop_fns=[])

  def test_all_stoppers_true_stops(self):
    """When all chained stoppers return True, should stop."""
    fn1 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=1.0
    )
    fn2 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=2.0
    )
    chained = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn1, fn2]
    )

    logits = jnp.full((2, 3, 4), -100.0).at[:, :, 0].set(100.0)
    step = _make_diffusion_step(xt=jnp.ones((2, 4, 1)), aux={'logits': logits})

    result = chained.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(jnp.all(result))

  def test_any_stopper_false_continues(self):
    """When one chained stopper returns False, should continue (AND logic)."""
    fn1 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=10.0
    )
    fn2 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=1e-5
    )
    chained = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn1, fn2]
    )

    logits = jnp.zeros((2, 3, 4))
    step = _make_diffusion_step(xt=jnp.ones((2, 3, 1)), aux={'logits': logits})

    result = chained.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(jnp.all(~result))

  def test_per_batch_chained(self):
    """Logical AND per batch element across multiple stoppers."""
    fn1 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.1
    )
    fn2 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.5
    )
    chained = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn1, fn2]
    )

    logits = jnp.array([
        [[100.0, -100.0, -100.0, -100.0]] * 3,
        [[0.0, 0.0, 0.0, 0.0]] * 3,
    ])
    step = _make_diffusion_step(xt=jnp.ones((2, 4, 1)), aux={'logits': logits})

    result = chained.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    chex.assert_shape(result, (2,))
    self.assertTrue(result[0])
    self.assertFalse(result[1])

  def test_order_does_not_matter(self):
    """Logical AND per batch element across multiple stoppers."""
    fn1 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.1
    )
    fn2 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=0.5
    )
    fn3 = diffusion_early_stopping.DiffusionEntropyEarlyStopFn(
        entropy_threshold=1.0
    )
    chained_1 = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn1, fn2, fn3]
    )
    chained_2 = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn1, fn3, fn2]
    )
    chained_3 = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn3, fn2, fn1]
    )

    logits = jnp.array([
        [[100.0, -100.0, -100.0, -100.0]] * 3,
        [[0.0, 0.0, 0.0, 0.0]] * 3,
    ])
    step = _make_diffusion_step(xt=jnp.ones((2, 4, 1)), aux={'logits': logits})

    result_1 = chained_1.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    result_2 = chained_2.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    result_3 = chained_3.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    chex.assert_trees_all_equal(result_1, result_2)
    chex.assert_trees_all_equal(result_2, result_3)
    chex.assert_trees_all_equal(result_1, result_3)

  def test_single_stopper_chain(self):
    """Works correctly with a single stopper in the chain."""
    fn = diffusion_early_stopping.DiffusionNoEarlyStopFn()
    chained = diffusion_early_stopping.DiffusionChainedEarlyStopFn(
        early_stop_fns=[fn]
    )
    step = _make_diffusion_step(xt=jnp.ones((4, 8, 1)))

    result = chained.should_stop(
        step=jnp.int32(0), current_step=step, previous_step=step
    )
    chex.assert_shape(result, (4,))
    self.assertTrue(jnp.all(~result))


if __name__ == '__main__':
  absltest.main()
