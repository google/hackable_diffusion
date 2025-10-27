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

"""Helper functions for different modules.

The file contains reusable NN components and factory functions for commonly used
operations in the architecture.
"""

import functools
import flax.linen as nn
from hackable_diffusion.lib.architecture import arch_typing
import jax
import jax.numpy as jnp


################################################################################
# MARK: Type Aliases
################################################################################

DownsampleFn = arch_typing.DownsampleFn
UpsampleFn = arch_typing.UpsampleFn
SkipConnectionFn = arch_typing.SkipConnectionFn
DownsampleType = arch_typing.DownsampleType
UpsampleType = arch_typing.UpsampleType
SkipConnectionMethod = arch_typing.SkipConnectionMethod

################################################################################
# MARK: Reusable NN components
################################################################################

kernel_init = nn.initializers.lecun_normal()
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3), padding="SAME")
ZerosConv3x3 = functools.partial(
    nn.Conv,
    kernel_size=(3, 3),
    padding="SAME",
    kernel_init=nn.initializers.zeros_init(),
    bias_init=nn.initializers.zeros_init(),
)
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1), padding="SAME")


################################################################################
# MARK: Factory Functions
################################################################################


def get_downsample_fn(downsample_type: DownsampleType) -> DownsampleFn:
  """Returns the downsampling function."""
  if downsample_type == DownsampleType.MAX_POOL:
    return lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
  elif downsample_type == DownsampleType.AVG_POOL:
    return lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
  else:
    raise ValueError(f"Unknown downsample type: {downsample_type}")


def get_upsample_fn(upsample_type: UpsampleType) -> UpsampleFn:
  """Returns the upsampling function."""
  return lambda x: jax.image.resize(
      x,
      (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
      method=str(upsample_type),
  )


def get_skip_connection_fn(
    skip_connection_method: SkipConnectionMethod,
) -> SkipConnectionFn:
  """Returns the skip connection function."""
  if skip_connection_method == SkipConnectionMethod.UNNORMALIZED_ADD:
    return lambda x, skip: x + skip
  elif skip_connection_method == SkipConnectionMethod.NORMALIZED_ADD:
    return lambda x, skip: (x + skip) / jnp.sqrt(2)
  else:
    raise ValueError(f"Unknown skip connection type: {skip_connection_method}")
