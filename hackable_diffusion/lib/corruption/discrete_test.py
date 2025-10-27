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

"""Tests for discrete corruption processes."""

from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import schedules
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: CategoricalProcessTest
################################################################################


class CategoricalProcessTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 5
    self.process = discrete.CategoricalProcess.uniform_process(
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
      process = discrete.CategoricalProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    elif process_type == 'uniform':
      process = discrete.CategoricalProcess.uniform_process(
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
    process = discrete.CategoricalProcess.masking_process(
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
      process = discrete.CategoricalProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    elif process_type == 'uniform':
      process = discrete.CategoricalProcess.uniform_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    else:
      raise ValueError(f'Unknown process type: {process_type}')
    time_0 = jnp.zeros((self.batch_size, self.seq_len, 1))
    xt_init, _ = process.corrupt(self.key, self.x0, time_0)
    self.assertTrue(jnp.array_equal(xt_init, self.x0))

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_corrupt_at_t1(self, process_type):
    # At t=1, alpha=0, so xt should be pure noise.
    if process_type == 'masking':
      process = discrete.CategoricalProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    elif process_type == 'uniform':
      process = discrete.CategoricalProcess.uniform_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
    else:
      raise ValueError(f'Unknown process type: {process_type}')
    time_final = jnp.ones((self.batch_size, self.seq_len, 1))
    key1, _ = jax.random.split(self.key)
    # Use a constant x0 to make it unlikely to be identical to noise.
    x0 = jnp.zeros(self.shape, dtype=jnp.int32)
    xt_final, _ = process.corrupt(key1, x0, time_final)
    # Check that corruption happened.
    if process_type == 'masking':
      mask_value = self.num_categories
      self.assertTrue(jnp.all(xt_final == jnp.ones_like(x0) * mask_value))
    elif process_type == 'uniform':
      bincount = jnp.bincount(xt_final.flatten())
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
      discrete.CategoricalProcess.uniform_process(
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
    factory = getattr(discrete.CategoricalProcess, factory_name)
    process = factory(schedule=self.schedule, num_categories=num_categories)
    self.assertEqual(
        process.process_num_categories, expected_process_num_categories
    )
    self.assertEqual(process.num_categories, num_categories)
    self.assertTrue(jnp.allclose(process.invariant_probs_vec, expected_probs))

  def test_factory_methods_raise_for_invalid_num_categories(self):
    with self.assertRaises(ValueError):
      discrete.CategoricalProcess.uniform_process(
          schedule=self.schedule, num_categories=0
      )
    with self.assertRaises(ValueError):
      discrete.CategoricalProcess.masking_process(
          schedule=self.schedule, num_categories=0
      )

  def test_symmetric_post_corruption(self):
    # Create a non-symmetric input
    x = jnp.array([[[[1], [2]], [[2], [3]]]], dtype=jnp.int32)
    post_corruption_fn = discrete.SymmetricPostCorruptionFn()
    projected_x = post_corruption_fn(x)
    # The symmetric projection takes the upper triangle and transpose it.
    # Note that we have 0 on the diagonal.
    expected_x = jnp.array([[[[0], [2]], [[2], [0]]]], dtype=jnp.int32)
    self.assertTrue(jnp.array_equal(projected_x, expected_x))

  def test_raises_for_non_4d_input(self):
    x_3d = jnp.ones((2, 2, 1), dtype=jnp.int32)
    post_corruption_fn = discrete.SymmetricPostCorruptionFn()
    with self.assertRaisesRegex(ValueError, 'Expected 4D input'):
      post_corruption_fn(x_3d)

  def test_raises_for_non_square_input(self):
    x_nonsquare = jnp.ones((1, 2, 3, 1), dtype=jnp.int32)
    post_corruption_fn = discrete.SymmetricPostCorruptionFn()
    with self.assertRaisesRegex(ValueError, 'Expected square input'):
      post_corruption_fn(x_nonsquare)

  @parameterized.named_parameters(
      ('masking', 'masking'),
      ('uniform', 'uniform'),
  )
  def test_is_masking(self, process_type):
    if process_type == 'masking':
      process = discrete.CategoricalProcess.masking_process(
          schedule=self.schedule, num_categories=self.num_categories
      )
      self.assertTrue(process.is_masking)
    elif process_type == 'uniform':
      process = discrete.CategoricalProcess.uniform_process(
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
      process = discrete.CategoricalProcess(
          schedule=self.schedule,
          invariant_probs=[0.0] * self.num_categories + [1.0],  # masking
          num_categories=self.num_categories,
      )
      self.assertTrue(process.is_masking)
    elif process_type == 'uniform':
      process = discrete.CategoricalProcess(
          schedule=self.schedule,
          invariant_probs=[1.0 / self.num_categories] * self.num_categories,
          num_categories=self.num_categories,
      )
      self.assertFalse(process.is_masking)
    else:
      raise ValueError(f'Unknown process type: {process_type}')


if __name__ == '__main__':
  absltest.main()
