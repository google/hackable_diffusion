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

"""Base classes and wrappers for noise processes."""

from __future__ import annotations

import dataclasses
from typing import Protocol

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree
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


################################################################################
# MARK: NestedProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedProcess(CorruptionProcess):
  """Wrapper for a pytree of noise schedules that is mapped over the data.

  Enables using different noise schedules for different input modalities.
  E.g. a gaussian schedule for the image and a categorical schedule for the
  labels.
  """

  processes: PyTree[CorruptionProcess]

  @typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,
  ) -> DataTree:
    """Sample from the invariant distribution."""
    return utils.tree_map_with_key(
        lambda k, process, data: process.sample_from_invariant(k, data),
        key,
        self.processes,
        data_spec,
    )

  @typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,
      time: TimeTree,
  ) -> tuple[DataTree, TargetInfoTree]:
    x0_structure = jax.tree.structure(x0)
    time_structure = jax.tree.structure(time)
    if x0_structure != time_structure:
      raise ValueError(
          f'x0 and time must have the same structure. Got: {x0_structure=} and'
          f' {time_structure=}'
      )
    xt_and_targets = utils.tree_map_with_key(
        lambda k, process, x, t: process.corrupt(k, x, t),
        key,
        self.processes,
        x0,
        time,
    )
    # Unzip the tree (from a tree of tuples to a tuple of trees)
    xt = jax.tree.map(
        lambda x0, xt_and_targets: xt_and_targets[0], x0, xt_and_targets
    )
    target_info = jax.tree.map(
        lambda x0, xt_and_targets: xt_and_targets[1], x0, xt_and_targets
    )
    return xt, target_info

  @typechecked
  def convert_predictions(
      self,
      prediction: TargetInfoTree,
      xt: DataTree,
      time: TimeTree,
  ) -> TargetInfoTree:
    """Convert the prediction to the target type."""
    return jax.tree.map(
        lambda process, pred, xt, time: process.convert_predictions(
            pred, xt, time
        ),
        self.processes,
        prediction,
        xt,
        time,
    )

  @typechecked
  def get_schedule_info(self, time: TimeTree) -> ScheduleInfoTree:
    """Get the schedule info for the given time."""
    return jax.tree.map(
        lambda process, t: process.get_schedule_info(t),
        self.processes,
        time,
    )
