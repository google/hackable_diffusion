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

"""Base classes and wrappers for noise processes."""

from __future__ import annotations

from typing import Protocol

from hackable_diffusion.lib import hd_typing

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey

DataTree = hd_typing.DataTree
TargetInfoTree = hd_typing.TargetInfoTree
TimeTree = hd_typing.TimeTree
ScheduleInfoTree = hd_typing.ScheduleInfoTree

################################################################################
# MARK: Protocols
################################################################################


class CorruptionProcess(Protocol):
  """Base class for all corruption processes (continuous and discrete)."""

  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,
      time: TimeTree,
  ) -> tuple[DataTree, TargetInfoTree]:
    """Corrupt x0 according to time, and return xt and targets info."""

  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,
  ) -> DataTree:
    """Sample from the invariant distribution."""

  def convert_predictions(
      self,
      prediction: TargetInfoTree,
      xt: DataTree,
      time: TimeTree,
  ) -> TargetInfoTree:
    """Convert the prediction to the target type."""

  def get_schedule_info(self, time: TimeTree) -> ScheduleInfoTree:
    """Get the schedule info for the given time."""
