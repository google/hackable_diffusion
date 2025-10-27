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

"""API for diffusion losses."""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.loss.base import DiffusionLoss
from hackable_diffusion.lib.loss.base import NestedDiffusionLoss
from hackable_diffusion.lib.loss.base import WeightFn
from hackable_diffusion.lib.loss.discrete import DiffusionCrossEntropyLoss
from hackable_diffusion.lib.loss.gaussian import compute_continuous_diffusion_loss
from hackable_diffusion.lib.loss.gaussian import NoWeightLoss
from hackable_diffusion.lib.loss.gaussian import SiD2Loss
# pylint: enable=g-importing-member
