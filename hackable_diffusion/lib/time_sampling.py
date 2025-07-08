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

"""Time samplers are used during training to sample random times (noise levels).

Usually, one time per example is sampled, but they are more flexible than that,
and support e.g. sampling different times for different modalities, or sampling
multiple different times for each example (e.g. a different noise level for each
frame in a video as in "History Guided Video Diffusion")


In general time samplers return a pytree of time arrays with the same structure
as the input data.
Each time array is a float array in self.time_range (which defaults to [0.0,
1.0]) with a shape broadcastable to the corresponding data array.

In the simplest case, the input data is a single Array["b h w c"] and time is
a single Float["b 1 1 1"].
But more complex cases including multiple modalities, or different time values
for parts of the data are also possible.
"""

from __future__ import annotations

import dataclasses
import math
import typing

from hackable_diffusion.lib import utils

import jax
import jax.numpy as jnp
import jaxtyping

# Type Aliases
Array = jaxtyping.Array
Float = jaxtyping.Float
Key = jaxtyping.Key
PyTree = jaxtyping.PyTree
Shaped = jaxtyping.Shaped

Self = typing.Self


class TimeSampler(typing.Protocol):
  """Time sampler protocol."""

  def __call__(
      self, key: Key, data_spec: PyTree[Shaped[Array, "*data"]]
  ) -> PyTree[Float[Array, "*#data"]]:
    """Returns a pytree of time arrays with the same structure as the data."""


#  _____ ___ __  __ _____      ____    _    __  __ ____  _     _____ ____  ____
# |_   _|_ _|  \/  | ____|    / ___|  / \  |  \/  |  _ \| |   | ____|  _ \/ ___|
#   | |  | || |\/| |  _|      \___ \ / _ \ | |\/| | |_) | |   |  _| | |_) \___ \
#   | |  | || |  | | |___      ___) / ___ \| |  | |  __/| |___| |___|  _ < ___)
#   |_| |___|_|  |_|_____|    |____/_/   \_\_|  |_|_|   |_____|_____|_| \_\____/
# MARK: UniformTimeSampler
@dataclasses.dataclass(kw_only=True, frozen=True)
class UniformTimeSampler(TimeSampler):
  """Uniform time sampler for a single data array.

  Sample time uniformly from the time_range (default [0.0, 1.0]).

  Attributes:
    axes: Which data axes to keep the shape of. Default is (0,) which means that
      the time array will have a shape of `(B, 1, 1, ...)`. This is the case
      e.g. for image diffusion where `data_spec` has shape `(B, h, w, c)`, and
      each image has a single time value, so the time array will have shape `(B,
      1, 1, 1)`.
    time_range: The range of times to sample from. Default is [0.0, 1.0].
  """

  axes: tuple[int, ...] = (0,)
  time_range: tuple[float, float] = (0.0, 1.0)

  def __call__(
      self, key: Key, data_spec: Shaped[Array, "*data"]
  ) -> Float[Array, "*#data"]:
    shape = utils.get_broadcastable_shape(data_spec.shape, self.axes)
    minval, maxval = self.time_range
    return jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval)

  @classmethod
  def from_safety_epsilon(cls, safety_epsilon: float, **kwargs) -> Self:
    """Returns a time sampler with a time range adjusted for safety."""
    return cls(
        time_range=(0.0 + safety_epsilon, 1.0 - safety_epsilon),
        **kwargs,
    )


# MARK: NestedTimeSampler


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedTimeSampler(TimeSampler):
  """Wrapper to support a nested pytree of time samplers.

  The structure of the samplers should match the structure of the data.

  Usage Example:
    ```
    time_sampler = NestedTimeSampler(
        samplers={
            "image": UniformTimeSampler(),
            "label": BetaTimeSampler(alpha=1.0, beta=1.0),
        }
    )
    ```

  Attributes:
    samplers: A pytree of time samplers matching the structure of the data.
  """

  samplers: PyTree[TimeSampler]

  def __call__(
      self, key: Key, data_spec: PyTree[Shaped[Array, "*data"]]
  ) -> PyTree[Float[Array, "*#data"]]:
    def _call_sampler(key, sampler, data_spec):
      return sampler(key, data_spec)

    return utils.tree_map_with_key(_call_sampler, key, self.samplers, data_spec)


# MARK: Specialized Samplers


@dataclasses.dataclass(kw_only=True, frozen=True)
class UniformStratifiedTimeSampler(TimeSampler):
  """Uniform stratified time sampler.

  See https://arxiv.org/abs/2107.00630 (I.1).

  Attributes:
    axes: Which data axes to keep the shape of. Default is (0,) which means each
      example in the batch will have a single time.
    time_range: The range of times to sample from. Default is [0.0, 1.0].
  """

  axes: tuple[int, ...] = (0,)
  time_range: tuple[float, float] = (0.0, 1.0)

  def __call__(
      self, key: Key, data_spec: Shaped[Array, "*data"]
  ) -> Float[Array, "*#data"]:
    shape = utils.get_broadcastable_shape(data_spec.shape, self.axes)
    tensor_dim = math.prod(shape)

    uniform_key, permute_key = jax.random.split(key)
    u = jax.random.uniform(uniform_key)
    t = (jnp.arange(tensor_dim) + u) / tensor_dim
    minval, maxval = self.time_range
    t = t * (maxval - minval) + minval
    p = jax.random.permutation(permute_key, tensor_dim)
    return t[p].reshape(shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class UnbalancedTimestepSampler(TimeSampler):
  """Unbalanced time sampler from the JointDiT paper.

  See https://arxiv.org/abs/2505.00482 (Section 3.1, and A.3).

  Attributes:
    key1: The key in the data_spec to use for the first time array.
    key2: The key in the data_spec to use for the second time array.
    s1: The scale factor for the first time array.
    s2: The scale factor for the second time array.
    p_equal: The probability of setting t2 = 1 - t1.
  """

  key1: str = "image"
  key2: str = "depth"

  s1: float = 3.1582
  s2: float = 0.25

  p_equal: float = 0.5

  def __call__(
      self, key: Key, data_spec: dict[str, Shaped[Array, "*data"]]
  ) -> dict[str, Float[Array, "*#data"]]:
    # Check that the keys match the data.
    if set(data_spec.keys()) != {self.key1, self.key2}:
      raise KeyError(
          f"Data keys {data_spec.keys()} do not match the keys specified in the"
          f" sampler {self.key1=} and {self.key2=}."
      )

    shape1 = utils.get_broadcastable_shape(data_spec[self.key1].shape, (0,))
    shape2 = utils.get_broadcastable_shape(data_spec[self.key2].shape, (0,))

    key1, key2, switch_key = jax.random.split(key, 3)

    z1 = jax.random.normal(key1, shape=shape1)
    f = jax.nn.sigmoid(z1) * self.s1 / (1 + (self.s1 - 1) * jax.nn.sigmoid(z1))

    z2 = jax.random.normal(key2, shape=shape2)
    g = jax.nn.sigmoid(z2) * self.s2 / (1 + (self.s2 - 1) * jax.nn.sigmoid(z2))

    # With probability p_equal, set g = 1 - f.
    equal_mask = jax.random.bernoulli(switch_key, p=self.p_equal, shape=shape1)
    g = jax.lax.select(equal_mask, 1 - f, g)
    return {self.key1: f, self.key2: g}
