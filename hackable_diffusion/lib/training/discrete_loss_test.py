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

"""Tests for Discrete loss functions."""

import chex
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.training import discrete_loss
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized


class DiscreteLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bsz = 4
    self.seq_len = 16
    self.vocab_size = 32
    self.key = jax.random.PRNGKey(42)

    key_logits, key_x0 = jax.random.split(self.key)

    time = jnp.linspace(0.1, 0.9, self.bsz)
    logits = jax.random.uniform(
        key_logits, (self.bsz, self.seq_len, self.vocab_size)
    )
    x0 = jax.random.randint(
        key_x0, (self.bsz, self.seq_len, 1), 0, self.vocab_size
    )
    self.time = jax_helpers.bcast_right(time, x0.ndim)
    self.preds = {'logits': logits}
    self.targets = {'x0': x0}
    self.schedule = schedules.LinearDiscreteSchedule()

  def test_diffusion_cross_entropy_loss_computation(self):
    """Tests base diffusion cross entropy loss computation."""

    loss_fn = lambda preds, targets, time: discrete_loss._compute_discrete_diffusion_loss(
        preds=preds,
        targets=targets,
        time=time,
        use_mask=False,
    )
    loss = loss_fn(self.preds, self.targets, self.time)

    # compute negative cross entropy from labels
    expected_loss = jnp.mean(
        jnp.sum(
            jax.nn.one_hot(self.targets['x0'][..., 0], self.vocab_size)
            * jax.nn.log_softmax(self.preds['logits']),
            axis=-1,
        ),
        axis=-1,  # average over sequence length
    )
    expected_loss = -1.0 * expected_loss

    self.assertTrue(jnp.allclose(loss, expected_loss, atol=1e-6))

    self.assertEqual(loss.shape, (self.bsz,))
    self.assertFalse(jnp.isnan(loss).any())

  @parameterized.named_parameters(
      ('normalize_by_mask', True),
      ('do_not_normalize_by_mask', False),
  )
  def test_masks_extract_individual_losses(self, normalize_by_mask: bool):
    logits_a = 0.3
    logits_b = 0.7
    logits = jnp.array([[(logits_a, logits_b) for _ in range(3)]])
    labels = jnp.array([[[0], [1], [0]]])

    masks_1 = jnp.array([[[1], [0], [0]]])
    masks_2 = jnp.array([[[0], [1], [0]]])
    masks_3 = jnp.array([[[0], [0], [1]]])

    masks = [masks_1, masks_2, masks_3]

    time = jnp.array([0.5])

    log_softmax_a = -jnp.log(
        jnp.exp(logits_a) / (jnp.exp(logits_a) + jnp.exp(logits_b))
    )
    log_softmax_b = -jnp.log(
        jnp.exp(logits_b) / (jnp.exp(logits_a) + jnp.exp(logits_b))
    )

    expected_individual_losses = [log_softmax_a, log_softmax_b, log_softmax_a]
    if not normalize_by_mask:
      expected_individual_losses = [l / 3.0 for l in expected_individual_losses]

    preds = {'logits': logits}
    loss = discrete_loss.NoWeightDiscreteLoss(
        use_mask=True, mask_key='test_mask', normalize_by_mask=normalize_by_mask
    )

    for exp_loss, mask in zip(expected_individual_losses, masks):
      targets = {'x0': labels, 'test_mask': mask}
      res = loss(preds=preds, targets=targets, time=time)
      chex.assert_trees_all_close(res, jnp.array([exp_loss]))

  @parameterized.named_parameters(
      ('normalize_by_mask', True),
      ('do_not_normalize_by_mask', False),
  )
  def tests_masks_no_effect_when_use_mask_is_false(
      self, normalize_by_mask: bool
  ):
    logits_a = 0.3
    logits_b = 0.7
    logits = jnp.array([[(logits_a, logits_b) for _ in range(3)]])
    labels = jnp.array([[[0], [1], [0]]])

    masks_1 = jnp.array([[[1], [0], [0]]])
    masks_2 = jnp.array([[[0], [1], [0]]])
    masks_3 = jnp.array([[[0], [0], [1]]])

    masks = [masks_1, masks_2, masks_3]

    time = jnp.array([0.5])

    log_softmax_a = -jnp.log(
        jnp.exp(logits_a) / (jnp.exp(logits_a) + jnp.exp(logits_b))
    )
    log_softmax_b = -jnp.log(
        jnp.exp(logits_b) / (jnp.exp(logits_a) + jnp.exp(logits_b))
    )
    expected_loss = (log_softmax_a + log_softmax_b + log_softmax_a) / 3.0

    preds = {'logits': logits}
    loss = discrete_loss.NoWeightDiscreteLoss(
        use_mask=False,
        mask_key='test_mask',
        normalize_by_mask=normalize_by_mask,
    )

    for mask in masks:
      targets = {'x0': labels, 'test_mask': mask}
      res = loss(preds=preds, targets=targets, time=time)
      chex.assert_trees_all_close(res, jnp.array([expected_loss]))

  @parameterized.named_parameters(
      ('normalize_by_mask', True),
      ('do_not_normalize_by_mask', False),
  )
  def test_mask_gives_expected_results(self, normalize_by_mask: bool):
    logits_a = 0.3
    logits_b = 0.7
    logits = jnp.array([[(logits_a, logits_b) for _ in range(3)]])
    labels = jnp.array([[[0], [1], [0]]])

    masks_1 = jnp.array([[[1], [0], [0]]])  # log_softmax_a
    masks_2 = jnp.array([[[0], [1], [0]]])  # log_softmax_b
    masks_3 = jnp.array([[[0], [0], [1]]])  # log_softmax_a
    masks_4 = jnp.array([[[1], [1], [1]]])  # 2 * log_softmax_a + log_softmax_b
    masks_5 = jnp.array([[[1], [1], [0]]])  # log_softmax_a + log_softmax_b
    masks_6 = jnp.array([[[1], [0], [1]]])  # 2 * log_softmax_a
    masks_7 = jnp.array([[[0], [1], [1]]])  # log_softmax_b + log_softmax_a

    masks = [masks_1, masks_2, masks_3, masks_4, masks_5, masks_6, masks_7]

    time = jnp.array([0.5])

    log_softmax_a = -jnp.log(
        jnp.exp(logits_a) / (jnp.exp(logits_a) + jnp.exp(logits_b))
    )
    log_softmax_b = -jnp.log(
        jnp.exp(logits_b) / (jnp.exp(logits_a) + jnp.exp(logits_b))
    )

    if normalize_by_mask:
      expected_losses = [
          log_softmax_a,
          log_softmax_b,
          log_softmax_a,
          (2.0 * log_softmax_a + log_softmax_b) / 3.0,
          (log_softmax_a + log_softmax_b) / 2.0,
          (2.0 * log_softmax_a) / 2.0,
          (log_softmax_b + log_softmax_a) / 2.0,
      ]
    else:
      expected_losses = [
          log_softmax_a / 3.0,
          log_softmax_b / 3.0,
          log_softmax_a / 3.0,
          (2.0 * log_softmax_a + log_softmax_b) / 3.0,
          (log_softmax_a + log_softmax_b) / 3.0,
          (2.0 * log_softmax_a) / 3.0,
          (log_softmax_b + log_softmax_a) / 3.0,
      ]

    preds = {'logits': logits}
    loss = discrete_loss.NoWeightDiscreteLoss(
        use_mask=True, mask_key='test_mask', normalize_by_mask=normalize_by_mask
    )

    for exp_loss, mask in zip(expected_losses, masks):
      targets = {'x0': labels, 'test_mask': mask}
      res = loss(preds=preds, targets=targets, time=time)
      chex.assert_trees_all_close(res, jnp.array([exp_loss]))


if __name__ == '__main__':
  absltest.main()
