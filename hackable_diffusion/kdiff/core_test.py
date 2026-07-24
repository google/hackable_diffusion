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

"""Super basic tests for core."""

import chex
import flax.linen as nn
from hackable_diffusion import hd
from hackable_diffusion.kdiff import core
import jax
import jax.numpy as jnp

from absl.testing import absltest


class DummyNetwork(nn.Module):

  @nn.compact
  def __call__(self, time, xt, conditioning, is_training):
    return {"v": jnp.zeros_like(xt)}


class DiffusionTest(absltest.TestCase):

  def test_output_structure_and_shapes(self):
    model = core.Diffusion(
        network=DummyNetwork(),
        corruption_process=hd.corruption.GaussianProcess(
            schedule=hd.corruption.RFSchedule(),
        ),
        time_sampler=hd.training.time_sampling.UniformTimeSampler(),
    )
    x0 = jnp.ones((2, 8, 8, 3))
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "default": jax.random.PRNGKey(1)},
        x0=x0,
        is_training_property=True,
    )

    out = model.apply(
        variables,
        x0=x0,
        rngs={"default": jax.random.PRNGKey(2)},
        is_training_property=True,
    )

    self.assertIsInstance(out, dict)
    self.assertContainsSubset(
        {"output", "target", "xt", "noise_info"}, out.keys()
    )
    chex.assert_shape(out["xt"], (2, 8, 8, 3))
    self.assertIsInstance(out["target"], dict)
    chex.assert_shape(out["target"]["epsilon"], (2, 8, 8, 3))
    self.assertIsInstance(out["output"], dict)
    chex.assert_shape(out["output"]["epsilon"], (2, 8, 8, 3))
    chex.assert_shape(out["output"]["x0"], (2, 8, 8, 3))
    self.assertIsInstance(out["noise_info"], dict)


class KauldronLossWrapperTest(absltest.TestCase):

  def test_stop_gradient_on_targets(self):
    def dummy_loss(preds, targets, time):
      del time  # Unused.
      return (preds["v"] - targets["v"]) ** 2

    preds = {"v": jnp.array([1.0, 2.0])}
    time = jnp.array([0.5, 0.5])
    targets_val = jnp.array([3.0, 4.0])

    def loss_fn(targets_val, stop_gradient):
      wrapper = core.KauldronLossWrapper(
          loss=dummy_loss,
          stop_gradient_on_targets=stop_gradient,
      )
      return jnp.sum(
          wrapper.get_values(
              preds=preds,
              targets={"v": targets_val},
              time=time,
          )
      )

    # When stop_gradient_on_targets is False, gradient wrt targets is non-zero.
    grad_no_stop = jax.grad(loss_fn, argnums=0)(targets_val, False)
    chex.assert_trees_all_close(grad_no_stop, -2.0 * (preds["v"] - targets_val))

    # When stop_gradient_on_targets is True, gradient wrt targets is zero.
    grad_stop = jax.grad(loss_fn, argnums=0)(targets_val, True)
    chex.assert_trees_all_close(grad_stop, jnp.zeros_like(targets_val))


if __name__ == "__main__":
  absltest.main()
