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

"""Tests for Discrete loss functions."""

from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.loss import discrete
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
    self.time = utils.bcast_right(time, x0.ndim)
    self.preds = {'logits': logits}
    self.targets = {'x0': x0}
    self.schedule = schedules.LinearDiscreteSchedule()

  @parameterized.named_parameters(
      (
          'with_weight_fn',
          lambda schedule, preds, targets, time: jnp.ones_like(time),
      ),
      ('without_weight_fn', None),
  )
  def test_diffusion_cross_entropy_loss_computation(self, weight_fn):
    """Tests loss can be computed with and without a weight function."""

    loss_fn = (
        lambda preds, targets, time: discrete.compute_discrete_diffusion_loss(
            preds=preds,
            targets=targets,
            time=time,
            schedule=self.schedule,
            use_mask=False,
            weight_fn=weight_fn,
        )
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
    if weight_fn:
      coeff = utils.flatten_non_batch_dims(
          weight_fn(
              schedule=self.schedule,
              preds=self.preds,
              targets=self.targets,
              time=self.time,
          )
      )[..., 0]
    else:
      coeff = 1.0
    expected_loss = -1.0 * coeff * expected_loss

    self.assertTrue(jnp.allclose(loss, expected_loss, atol=1e-6))

    self.assertEqual(loss.shape, (self.bsz,))
    self.assertFalse(jnp.isnan(loss).any())


if __name__ == '__main__':
  absltest.main()
