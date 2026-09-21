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

"""Tests for hd_typing annotations and runtime ktyping checks."""

from hackable_diffusion.lib import hd_typing
import jax.numpy as jnp
import kauldron.ktyping as kt

from absl.testing import absltest

Array = hd_typing.Array
Bool = hd_typing.Bool
DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
Float = hd_typing.Float
Int = hd_typing.Int
LossOutput = hd_typing.LossOutput
PyTree = hd_typing.PyTree
ScheduleInfoTree = hd_typing.ScheduleInfoTree
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray
TimeTree = hd_typing.TimeTree


@hd_typing.typechecked
def _annotated_fn(
    x: DataArray,
    t: TimeArray,
    mask: Bool['*batch'],
    labels: Int['batch 1'],
) -> tuple[Float['batch *#data_shape'], TargetInfo, LossOutput]:
  del mask, labels
  out = x + t
  return out, {'x0': out}, jnp.mean(out, axis=tuple(range(1, out.ndim)))


@hd_typing.typechecked
def _annotated_tree_fn(
    tree: DataTree,
    time_tree: TimeTree,
    meta: PyTree[Float['*batch']],
) -> tuple[DataTree, ScheduleInfoTree]:
  del time_tree, meta
  return tree, {'a': {'time': tree['a']}}


class HdTypingTest(absltest.TestCase):

  def test_typechecked_annotations_valid(self):
    x = jnp.ones((2, 4, 4, 3), dtype=jnp.float32)
    t = jnp.ones((2, 1, 1, 1), dtype=jnp.float32)
    mask = jnp.array([True, False])
    labels = jnp.zeros((2, 1), dtype=jnp.int32)

    out, target_info, loss = _annotated_fn(x, t, mask, labels)
    self.assertEqual(out.shape, (2, 4, 4, 3))
    self.assertEqual(target_info['x0'].shape, (2, 4, 4, 3))
    self.assertEqual(loss.shape, (2,))

  def test_typechecked_annotations_reject_mismatch(self):
    x = jnp.ones((2, 4, 4, 3), dtype=jnp.float32)
    t = jnp.ones((2, 1, 1, 1), dtype=jnp.float32)
    bad_mask = jnp.array([True, False, True])
    labels = jnp.zeros((2, 1), dtype=jnp.int32)

    with self.assertRaises(kt.KTypeCheckError):
      _annotated_fn(x, t, bad_mask, labels)

  def test_typechecked_pytree_annotations(self):
    tree = {'a': jnp.ones((2, 3), dtype=jnp.float32)}
    time_tree = {'a': jnp.ones((2, 1), dtype=jnp.float32)}
    meta = {'w': jnp.ones((2,), dtype=jnp.float32)}

    out_tree, sched_tree = _annotated_tree_fn(tree, time_tree, meta)
    self.assertEqual(out_tree['a'].shape, (2, 3))
    self.assertEqual(sched_tree['a']['time'].shape, (2, 3))


if __name__ == '__main__':
  absltest.main()
