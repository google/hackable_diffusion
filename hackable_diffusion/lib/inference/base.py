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

"""Base inference function protocol."""

from typing import Protocol
from hackable_diffusion.lib import hd_typing

################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
TimeTree = hd_typing.TimeTree
TargetInfoTree = hd_typing.TargetInfoTree

################################################################################
# MARK: InferenceFn
################################################################################


class InferenceFn(Protocol):
  """A protocol for an inference function.

  The InferenceFn is responsible for predicting the clean data `x0` from the
  noisy input `xt`. It also predicts related quantities such as ['eps', 'score',
  'velocity', 'v'] in the case of a Gaussian diffusion model. It can also take
  into account guidance and/or conditioning.

  The InferenceFn is created by the create_inference_fn function.
  """

  def __call__(
      self, time: TimeTree, xt: DataTree, conditioning: Conditioning | None
  ) -> TargetInfoTree:
    ...
