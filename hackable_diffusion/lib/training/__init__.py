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

"""API for diffusion losses."""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.hd_api import DiffusionLoss
from hackable_diffusion.lib.training import time_sampling
from hackable_diffusion.lib.training.discrete_loss import MD4Loss
from hackable_diffusion.lib.training.discrete_loss import NoWeightDiscreteLoss
from hackable_diffusion.lib.training.gaussian_loss import NoWeightGaussianLoss
from hackable_diffusion.lib.training.gaussian_loss import SiD2Loss
from hackable_diffusion.lib.training.time_sampling import LogitNormalTimeSampler
from hackable_diffusion.lib.training.time_sampling import TimeSampler
from hackable_diffusion.lib.training.time_sampling import UnbalancedTimestepSampler
from hackable_diffusion.lib.training.time_sampling import UniformStratifiedTimeSampler
from hackable_diffusion.lib.training.time_sampling import UniformTimeSampler

# pylint: enable=g-importing-member
