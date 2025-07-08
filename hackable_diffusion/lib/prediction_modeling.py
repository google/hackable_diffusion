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

"""Provides wrapper and utility for the prediction model.

This encompases:
  1. Helpers to convert the output of the prediction (`x0`) into the various
     formats (score, velocity, etc.).
  2. Prediction model wrapping the backbone model (including conditioning and
      guidance when needed), the prediction model always returns at least x0.
"""

import dataclasses

from hackable_diffusion.lib import denoising
from hackable_diffusion.lib import noise_scheduling
import jax.numpy as jnp
import jaxtyping


# Type Aliases
Float = jaxtyping.Float
PyTree = jaxtyping.PyTree
Array = jaxtyping.Array
PRNGKeyArray = jaxtyping.PRNGKeyArray

GaussianNoiseSchedule = noise_scheduling.GaussianNoiseSchedule


# __  _____       _    ____    _    ____ _____ _____ ____  ____
# \ \/ / _ \     / \  |  _ \  / \  |  _ \_   _| ____|  _ \/ ___|
#  \  / | | |   / _ \ | | | |/ _ \ | |_) || | |  _| | |_) \___ \
#  /  \ |_| |  / ___ \| |_| / ___ \|  __/ | | | |___|  _ < ___) |
# /_/\_\___/  /_/   \_\____/_/   \_\_|    |_| |_____|_| \_\____/
# MARK: X0 Adapters
def x0_to_score(
    process: GaussianNoiseSchedule, x0: Array, t: Float, xt: Array
) -> Array:
  """Converts a prediction of the clean data (`x0`) into the score function."""
  alpha, sigma = process.alpha(t), process.sigma(t)
  return (alpha * x0 - xt) * jnp.reciprocal(sigma**2)


def get_score(
    process: GaussianNoiseSchedule, prediction: PyTree, t: Float, xt: Array
) -> Array:
  return prediction.get('score', x0_to_score(process, prediction['x0'], t, xt))


#  ____  ____  _____ ____       __  __  ___  ____  _____ _     ____
# |  _ \|  _ \| ____|  _ \     |  \/  |/ _ \|  _ \| ____| |   / ___|
# | |_) | |_) |  _| | | | |    | |\/| | | | | | | |  _| | |   \___ \
# |  __/|  _ <| |___| |_| |    | |  | | |_| | |_| | |___| |___ ___) |
# |_|   |_| \_\_____|____/     |_|  |_|\___/|____/|_____|_____|____/
# MARK: Prediction Model
@dataclasses.dataclass(frozen=True, kw_only=True)
class IdentityPredictionModel(denoising.PredictionModel):
  """A prediction model that returns the input as the prediction.

  This is useful for testing and debugging purposes.
  """

  def predict(
      self, xt: PyTree, conditioning: PyTree, t: PyTree, rng: PRNGKeyArray
  ) -> PyTree:
    return dict(x0=xt)  # the prediction is the same as the input
