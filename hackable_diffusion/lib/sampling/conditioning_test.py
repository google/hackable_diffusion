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

"""Tests for conditioning update functions."""

import chex
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib.sampling import conditioning
import jax
import jax.numpy as jnp

from absl.testing import absltest


def _make_step_info() -> hd_api.StepInfo:
  return hd_api.StepInfo(
      step=jnp.int32(0),
      time=jnp.float32(0.5),
      rng=jax.random.PRNGKey(0),
  )


class PropagateSelfConditioningFnTest(absltest.TestCase):
  """Tests for PropagateSelfConditioningFn."""

  def test_single_modal(self):
    logits = jnp.ones((2, 4, 10))
    step_carry = hd_api.DiffusionStep(
        xt=jnp.zeros((2, 4, 1)),
        step_info=_make_step_info(),
        aux={'logits': logits},
    )
    fn = conditioning.PropagateSelfConditioningFn()
    base_cond = {'label': jnp.array([1, 2])}
    updated = fn(base_cond, step_carry)

    self.assertIn('label', updated)
    self.assertIn('self_conditioning_logits', updated)
    chex.assert_trees_all_close(updated['self_conditioning_logits'], logits)
    self.assertNotIn('self_conditioning_logits', base_cond)

  def test_none_conditioning(self):
    logits = jnp.ones((2, 4, 10))
    step_carry = hd_api.DiffusionStep(
        xt=jnp.zeros((2, 4, 1)),
        step_info=_make_step_info(),
        aux={'logits': logits},
    )
    fn = conditioning.PropagateSelfConditioningFn()
    updated = fn(None, step_carry)

    self.assertEqual(list(updated.keys()), ['self_conditioning_logits'])
    chex.assert_trees_all_close(updated['self_conditioning_logits'], logits)

  def test_custom_key_and_multimodal_pytree(self):
    step_carry = {
        'mod1': hd_api.DiffusionStep(
            xt=jnp.zeros((2, 4, 1)),
            step_info=_make_step_info(),
            aux={'logits': jnp.ones((2, 4, 10))},
        ),
        'mod2': hd_api.DiffusionStep(
            xt=jnp.zeros((2, 1)),
            step_info=_make_step_info(),
            aux={'logits': jnp.full((2, 5), 2.0)},
        ),
    }
    fn = conditioning.PropagateSelfConditioningFn(
        self_conditioning_key='custom_sc_key'
    )
    updated = fn({'label': jnp.array([1, 2])}, step_carry)

    self.assertIn('label', updated)
    self.assertIn('custom_sc_key', updated)
    chex.assert_trees_all_close(
        updated['custom_sc_key']['mod1'], jnp.ones((2, 4, 10))
    )
    chex.assert_trees_all_close(
        updated['custom_sc_key']['mod2'], jnp.full((2, 5), 2.0)
    )


if __name__ == '__main__':
  absltest.main()
