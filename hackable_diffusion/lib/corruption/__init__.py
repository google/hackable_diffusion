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

"""API for corruption processes."""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.corruption.base import CorruptionProcess
from hackable_diffusion.lib.corruption.base import NestedProcess
from hackable_diffusion.lib.corruption.discrete import CategoricalProcess
from hackable_diffusion.lib.corruption.discrete import IdentityPostCorruptionFn
from hackable_diffusion.lib.corruption.discrete import PostCorruptionFn
from hackable_diffusion.lib.corruption.discrete import SymmetricPostCorruptionFn
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.schedules import CosineDiscreteSchedule
from hackable_diffusion.lib.corruption.schedules import CosineSchedule
from hackable_diffusion.lib.corruption.schedules import DiscreteSchedule
from hackable_diffusion.lib.corruption.schedules import GaussianSchedule
from hackable_diffusion.lib.corruption.schedules import GeometricDiscreteSchedule
from hackable_diffusion.lib.corruption.schedules import GeometricSchedule
from hackable_diffusion.lib.corruption.schedules import InverseCosineSchedule
from hackable_diffusion.lib.corruption.schedules import LinearDiffusionSchedule
from hackable_diffusion.lib.corruption.schedules import LinearDiscreteSchedule
from hackable_diffusion.lib.corruption.schedules import PolynominalDiscreteSchedule
from hackable_diffusion.lib.corruption.schedules import RFSchedule
from hackable_diffusion.lib.corruption.schedules import Schedule
from hackable_diffusion.lib.corruption.schedules import ShiftedSchedule
from hackable_diffusion.lib.corruption.schedules import SquareCosineDiscreteSchedule
from hackable_diffusion.lib.corruption.simplicial import SimplicialProcess
from hackable_diffusion.lib.corruption.simplicial import SimplicialSchedule
# pylint: enable=g-importing-member
