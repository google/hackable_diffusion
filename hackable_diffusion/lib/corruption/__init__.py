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

"""API for corruption processes."""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.corruption.discrete import CategoricalProcess
from hackable_diffusion.lib.corruption.discrete import CosineDiscreteSchedule
from hackable_diffusion.lib.corruption.discrete import DiscreteSchedule
from hackable_diffusion.lib.corruption.discrete import GeometricDiscreteSchedule
from hackable_diffusion.lib.corruption.discrete import IdentityPostCorruptionFn
from hackable_diffusion.lib.corruption.discrete import LinearDiscreteSchedule
from hackable_diffusion.lib.corruption.discrete import PolynomialDiscreteSchedule
from hackable_diffusion.lib.corruption.discrete import PostCorruptionFn
from hackable_diffusion.lib.corruption.discrete import SquareCosineDiscreteSchedule
from hackable_diffusion.lib.corruption.discrete import SymmetricPostCorruptionFn
from hackable_diffusion.lib.corruption.gaussian import CosineSchedule
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.gaussian import GaussianSchedule
from hackable_diffusion.lib.corruption.gaussian import GeometricSchedule
from hackable_diffusion.lib.corruption.gaussian import InverseCosineSchedule
from hackable_diffusion.lib.corruption.gaussian import LinearDiffusionSchedule
from hackable_diffusion.lib.corruption.gaussian import RFSchedule
from hackable_diffusion.lib.corruption.gaussian import ShiftedSchedule
from hackable_diffusion.lib.corruption.riemannian import LinearRiemannianSchedule
from hackable_diffusion.lib.corruption.riemannian import RiemannianProcess
from hackable_diffusion.lib.corruption.riemannian import RiemannianSchedule
from hackable_diffusion.lib.corruption.simplicial import CosineSimplicialSchedule
from hackable_diffusion.lib.corruption.simplicial import GeometricSimplicialSchedule
from hackable_diffusion.lib.corruption.simplicial import IdentitySimplicialPostCorruptionFn
from hackable_diffusion.lib.corruption.simplicial import LinearSimplicialSchedule
from hackable_diffusion.lib.corruption.simplicial import PolynomialSimplicialSchedule
from hackable_diffusion.lib.corruption.simplicial import SimplicialPostCorruptionFn
from hackable_diffusion.lib.corruption.simplicial import SimplicialProcess
from hackable_diffusion.lib.corruption.simplicial import SimplicialSchedule
from hackable_diffusion.lib.corruption.simplicial import SquareCosineSimplicialSchedule
from hackable_diffusion.lib.corruption.simplicial import SymmetricSimplicialPostCorruptionFn
from hackable_diffusion.lib.hd_api import CorruptionProcess
from hackable_diffusion.lib.hd_api import CorruptionSchedule as Schedule

# pylint: enable=g-importing-member
