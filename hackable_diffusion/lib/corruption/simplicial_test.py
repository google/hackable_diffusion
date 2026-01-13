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

"""Tests for simplicial corruption processes.

The main difference between categorical and simplicial corruption is that xt is
not of shape [B, T, 1] but [B, T, K]. Instead of tracking a realization of a
certain categorical distribution, we track a realization of a certain Dirichlet
distribution. Therefore, to obtain a token, we must convert the probability
distribution to a token. Here we use the argmax function (we could have also
used categorical sampling).

Apart from this crucial difference, the tests are mostly identical with the
difference that we do not allow for post-corruption projection in the simplicial
corruption process.
"""

from hackable_diffusion.lib.corruption import schedules
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
    self.schedule = schedules.LinearDiscreteSchedule()
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
      ('masking_high', 'masking', simplicial.SamplingPrecisionMode.HIGH),
      ('masking_low', 'masking', simplicial.SamplingPrecisionMode.LOW),
      ('uniform_high', 'uniform', simplicial.SamplingPrecisionMode.HIGH),
      ('uniform_low', 'uniform', simplicial.SamplingPrecisionMode.LOW),
  )
  def test_corrupt_at_t1_with_mode(self, process_type, mode):
    # At t=1, alpha=0, so xt should be pure noise.
    if process_type == 'masking':
      process = simplicial.SimplicialProcess(
          schedule=self.schedule,
          invariant_probs=[0.0] * self.num_categories + [1.0],
          num_categories=self.num_categories,
          mode=mode,
      )
    elif process_type == 'uniform':
      process = simplicial.SimplicialProcess(
          schedule=self.schedule,
          invariant_probs=[1.0 / self.num_categories] * self.num_categories,
          num_categories=self.num_categories,
          mode=mode,
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

  def test_non_valid_unused_mask_value(self):
    with self.assertRaises(ValueError):
      simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule,
          num_categories=self.num_categories,
          unused_mask_value=0,
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
    with self.assertRaises(ValueError):
      simplicial.SimplicialProcess.uniform_process(
          schedule=self.schedule, num_categories=0
      )
    with self.assertRaises(ValueError):
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


if __name__ == '__main__':
  absltest.main()
