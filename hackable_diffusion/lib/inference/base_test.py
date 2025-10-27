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

"""Tests for base inference functions."""

import chex
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.inference import base
import jax.numpy as jnp

from absl.testing import absltest


################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
TargetInfo = hd_typing.TargetInfo
TimeTree = hd_typing.TimeTree

GaussianProcess = gaussian.GaussianProcess
InferenceFn = base.InferenceFn

################################################################################
# MARK: Helper functions
################################################################################


def get_dummy_inference_fn(process: GaussianProcess) -> InferenceFn:

  def dummy_inference_fn(
      xt: DataTree, conditioning: Conditioning, time: TimeTree
  ) -> TargetInfo:
    """Dummy inference function."""
    del conditioning  # unused
    return process.convert_predictions(
        prediction={"x0": xt},
        xt=xt,
        time=time,
    )

  return dummy_inference_fn


################################################################################
# MARK: Tests
################################################################################


class IdentityInferenceFnTest(absltest.TestCase):

  def test_predict(self):
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    dummy_inference_fn = get_dummy_inference_fn(process=process)

    chex.assert_trees_all_equal(
        dummy_inference_fn(
            xt=jnp.eye(2),
            conditioning={},
            time=jnp.array([0.5, 0.5]),
        ),
        dict(
            process.convert_predictions(
                prediction={"x0": jnp.eye(2)},
                xt=jnp.eye(2),
                time=jnp.array([0.5, 0.5]),
            )
        ),
    )


if __name__ == "__main__":
  absltest.main()
