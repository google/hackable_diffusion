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

"""Test for base losses."""

import chex
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.loss import base as base_loss
from hackable_diffusion.lib.loss import discrete as discrete_loss
from hackable_diffusion.lib.loss import gaussian as gaussian_loss
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


class BaseLossTest(parameterized.TestCase):
  """Tests for the base losses."""

  def test_nested_loss(self):
    """Test that the nested loss is correctly applied."""

    schedule_discrete = schedules.SquareCosineDiscreteSchedule()
    loss_fn_continuous = gaussian_loss.NoWeightLoss()
    loss_fn_discrete = discrete_loss.DiffusionCrossEntropyLoss(
        schedule=schedule_discrete
    )
    loss_fn = base_loss.NestedDiffusionLoss(
        losses={
            "data_continuous": loss_fn_continuous,
            "data_discrete": loss_fn_discrete,
        }
    )

    # dummy time, targets and outputs
    time = {
        "data_continuous": jnp.ones((1,)) * 0.5,
        "data_discrete": jnp.ones((1,)) * 0.5,
    }
    targets = {
        "data_continuous": {"x0": jnp.ones((1, 32, 32, 3))},
        "data_discrete": {
            "x0": jnp.ones((1, 128, 1), dtype=jnp.int32),
            "logits": jnp.ones((1, 128, 64), dtype=jnp.int32),
        },
    }
    output = {
        "data_continuous": {"x0": jnp.ones((1, 32, 32, 3))},
        "data_discrete": {"logits": jnp.ones((1, 128, 64))},
    }

    loss_output = loss_fn(preds=output, targets=targets, time=time)

    chex.assert_trees_all_equal_structs(loss_output, time)


if __name__ == "__main__":
  absltest.main()
