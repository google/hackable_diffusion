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

"""Unit tests for Tier-1 contracts and operand-parametrized types in hd_api."""

import dataclasses
from typing import Any, TypeAlias

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib import multimodal
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import typeguard

from absl.testing import absltest

Batch: TypeAlias = hd_api.Batch
Tree: TypeAlias = multimodal.Tree
PRNGKey: TypeAlias = hd_api.PRNGKey
typechecked = jax_helpers.typechecked

_CHECK_ERRORS = (
    typeguard.TypeCheckError,
    jt.TypeCheckError,
    TypeError,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearTestSchedule(hd_api.CorruptionSchedule[Batch]):

  @typechecked
  def evaluate(
      self, time: hd_api.Time[Batch]
  ) -> hd_api.ScheduleInfo[Batch]:
    return {
        'time': time,
        'alpha': 1.0 - time,
        'sigma': time,
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearTestProcess(hd_api.CorruptionProcess[Batch]):
  schedule: LinearTestSchedule = dataclasses.field(
      default_factory=LinearTestSchedule
  )

  @typechecked
  def sample_from_invariant(
      self, key: PRNGKey, data_spec: hd_api.Data[Batch]
  ) -> hd_api.Data[Batch]:
    return jax.random.normal(key, data_spec.shape, data_spec.dtype)

  @typechecked
  def corrupt(
      self, key: PRNGKey, x0: hd_api.Data[Batch], time: hd_api.Time[Batch]
  ) -> tuple[hd_api.Data[Batch], hd_api.TargetInfo[Batch]]:
    eps = self.sample_from_invariant(key, x0)
    info = self.schedule.evaluate(time)
    xt = info['alpha'] * x0 + info['sigma'] * eps
    return xt, {'x0': x0, 'eps': eps}

  @typechecked
  def convert_predictions(
      self,
      prediction: hd_api.TargetInfo[Batch],
      xt: hd_api.Data[Batch],
      time: hd_api.Time[Batch],
  ) -> hd_api.TargetInfo[Batch]:
    del xt, time
    return {**prediction}


@dataclasses.dataclass(frozen=True, kw_only=True)
class NestedTestSchedule(hd_api.CorruptionSchedule[Tree]):
  schedules: jt.PyTree[hd_api.CorruptionSchedule[Batch]]

  @typechecked
  def evaluate(
      self, time: hd_api.Time[Tree]
  ) -> hd_api.ScheduleInfo[Tree]:
    return jax.tree.map(lambda s, t: s.evaluate(t), self.schedules, time)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NestedTestProcess(hd_api.CorruptionProcess[Tree]):
  processes: jt.PyTree[hd_api.CorruptionProcess[Batch]]

  @property
  def schedule(self) -> hd_api.CorruptionSchedule[Tree]:
    return NestedTestSchedule(
        schedules=jax.tree.map(lambda p: p.schedule, self.processes)
    )

  @typechecked
  def sample_from_invariant(
      self, key: PRNGKey, data_spec: hd_api.Data[Tree]
  ) -> hd_api.Data[Tree]:
    return jax_helpers.tree_map_with_key(
        lambda k, p, d: p.sample_from_invariant(k, d),
        key,
        self.processes,
        data_spec,
    )

  @typechecked
  def corrupt(
      self, key: PRNGKey, x0: hd_api.Data[Tree], time: hd_api.Time[Tree]
  ) -> tuple[hd_api.Data[Tree], hd_api.TargetInfo[Tree]]:
    pairs = jax_helpers.tree_map_with_key(
        lambda k, p, x, t: p.corrupt(k, x, t),
        key,
        self.processes,
        x0,
        time,
    )
    xt_out = jax.tree.map(lambda _, pair: pair[0], x0, pairs)
    pred_out = jax.tree.map(lambda _, pair: pair[1], x0, pairs)
    return xt_out, pred_out

  @typechecked
  def convert_predictions(
      self,
      prediction: hd_api.TargetInfo[Tree],
      xt: hd_api.Data[Tree],
      time: hd_api.Time[Tree],
  ) -> hd_api.TargetInfo[Tree]:
    return jax.tree.map(
        lambda p, pred, x, t: p.convert_predictions(pred, x, t),
        self.processes,
        prediction,
        xt,
        time,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MSETestLoss(hd_api.DiffusionLoss[Batch]):

  @typechecked
  def __call__(
      self,
      preds: hd_api.TargetInfo[Batch],
      targets: hd_api.TargetInfo[Batch],
      time: hd_api.Time[Batch],
  ) -> hd_api.Loss[Batch]:
    del time
    err = jnp.square(preds['x0'] - targets['x0'])
    return jnp.mean(jax_helpers.flatten_non_batch_dims(err), axis=-1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EulerTestStep(hd_api.SamplerStep[Batch]):

  @typechecked
  def initialize(
      self,
      initial_noise: hd_api.Data[Batch],
      initial_step_info: hd_api.StepInfo[Batch],
  ) -> hd_api.DiffusionStep[Batch]:
    return hd_api.DiffusionStep(
        xt=initial_noise, step_info=initial_step_info, aux={}
    )

  @typechecked
  def update(
      self,
      prediction: hd_api.TargetInfo[Batch],
      current_step: hd_api.DiffusionStep[Batch],
      next_step_info: hd_api.StepInfo[Batch],
  ) -> hd_api.DiffusionStep[Batch]:
    dt = current_step.step_info.time - next_step_info.time
    xt_next = current_step.xt - dt * (current_step.xt - prediction['x0'])
    return hd_api.DiffusionStep(
        xt=xt_next, step_info=next_step_info, aux={}
    )

  @typechecked
  def finalize(
      self,
      prediction: hd_api.TargetInfo[Batch],
      current_step: hd_api.DiffusionStep[Batch],
      last_step_info: hd_api.StepInfo[Batch],
  ) -> hd_api.DiffusionStep[Batch]:
    return self.update(prediction, current_step, last_step_info)


class HdApiContractsTest(absltest.TestCase):

  def test_batch_process_and_loss_shape_checks(self):
    sched: hd_api.CorruptionSchedule[Batch] = LinearTestSchedule()
    proc: hd_api.CorruptionProcess[Batch] = LinearTestProcess()
    loss_fn: hd_api.DiffusionLoss[Batch] = MSETestLoss()
    key = jax.random.key(0)
    x0 = jnp.ones((3, 4, 2))
    t_ok = jnp.full((3, 1, 1), 0.25)

    info = sched.evaluate(t_ok)
    self.assertEqual(info['time'].shape, (3, 1, 1))
    xt, target = proc.corrupt(key, x0, t_ok)
    self.assertEqual(xt.shape, (3, 4, 2))
    loss = loss_fn({'x0': jnp.zeros_like(x0)}, target, t_ok)
    self.assertEqual(loss.shape, (3,))
    np.testing.assert_allclose(loss, jnp.ones((3,)), atol=1e-6)

    # Per-token / per-frame time (e.g. Diffusion Forcing) is valid.
    t_per_token = jnp.linspace(0.1, 0.9, 12).reshape((3, 4, 1))
    xt_tok, _ = proc.corrupt(key, x0, t_per_token)
    self.assertEqual(xt_tok.shape, (3, 4, 2))

    # Non-broadcastable spatial dim on `time` fails @typechecked.
    with self.assertRaises(_CHECK_ERRORS):
      proc.corrupt(key, x0, jnp.full((3, 5, 1), 0.25))

    # Wrong batch size on `time` fails @typechecked.
    with self.assertRaises(_CHECK_ERRORS):
      proc.corrupt(key, x0, jnp.full((7, 1, 1), 0.25))

    # 1D `time` fails rank/broadcast check against 3D `x0`.
    with self.assertRaises(_CHECK_ERRORS + (ValueError,)):
      proc.corrupt(key, x0, jnp.full((3,), 0.25))

    # Wrong batch size on preds/targets vs `time` fails on Loss[Batch].
    with self.assertRaises(_CHECK_ERRORS):
      loss_fn(
          {'x0': jnp.zeros((7, 4, 2))},
          {'x0': jnp.zeros((7, 4, 2))},
          t_ok,
      )

  def test_open_typed_dict_schedule_and_target_contracts(self):
    @typechecked
    def consume_schedule_and_targets(
        info: hd_api.ScheduleInfo[Batch],
        targets: hd_api.TargetInfo[Batch],
    ) -> None:
      del info, targets

    t = jnp.full((3, 1, 1), 0.5)
    # On-the-fly schedule keys ('alpha_dot') and discrete/mask target keys
    # ('logits' with vocab dim K=10 and boolean 'is_corrupted') are accepted.
    consume_schedule_and_targets(
        {'time': t, 'alpha': 1.0 - t, 'alpha_dot': -jnp.ones_like(t)},
        {
            'x0': jnp.zeros((3, 8, 1), dtype=jnp.int32),
            'logits': jnp.zeros((3, 8, 10), dtype=jnp.float32),
            'is_corrupted': jnp.ones((3, 8, 1), dtype=jnp.bool_),
        },
    )

    # Non-dict value passed as ScheduleInfo fails runtime @typechecked.
    bad_info: Any = [('time', t)]
    with self.assertRaises(_CHECK_ERRORS):
      consume_schedule_and_targets(bad_info, {'x0': jnp.zeros((3, 8, 1))})

  def test_tree_operand_heterogeneous_modalities(self):
    nested: hd_api.CorruptionProcess[Tree] = NestedTestProcess(
        processes={'img': LinearTestProcess(), 'seq': LinearTestProcess()}
    )
    key = jax.random.key(1)
    x0 = {'img': jnp.ones((2, 4, 4, 3)), 'seq': jnp.zeros((2, 8, 1))}
    t_ok = {
        'img': jnp.full((2, 1, 1, 1), 0.3),
        'seq': jnp.full((2, 1, 1), 0.7),
    }
    sched_info = nested.schedule.evaluate(t_ok)
    self.assertEqual(sched_info['img']['alpha'].shape, (2, 1, 1, 1))
    xt, preds = nested.corrupt(key, x0, t_ok)
    self.assertEqual(xt['img'].shape, (2, 4, 4, 3))
    self.assertEqual(preds['seq']['x0'].shape, (2, 8, 1))

    # Mismatched PyTree structure between x0 and time fails @typechecked.
    with self.assertRaises(_CHECK_ERRORS):
      nested.corrupt(key, x0, {'img': jnp.full((2, 1, 1, 1), 0.3)})

    # Mismatched batch dimension across modalities fails @typechecked.
    t_bad_batch = {
        'img': jnp.full((2, 1, 1, 1), 0.3),
        'seq': jnp.full((5, 1, 1), 0.7),
    }
    with self.assertRaises(_CHECK_ERRORS):
      nested.corrupt(key, x0, t_bad_batch)

  def test_step_containers_jit_scan_and_frozen(self):
    stepper: hd_api.SamplerStep[Batch] = EulerTestStep()
    ts = jnp.linspace(1.0, 0.0, 4)[:, None, None] * jnp.ones((1, 2, 1))
    rngs = jax.random.split(jax.random.key(2), 4)
    init_info = hd_api.StepInfo(step=jnp.asarray(0), time=ts[0], rng=rngs[0])
    init_step = stepper.initialize(jnp.ones((2, 3)), init_info)

    with self.assertRaises(dataclasses.FrozenInstanceError):
      init_step.xt = jnp.zeros((2, 3))  # pyrefly: ignore[read-only]

    @jax.jit
    def scan_steps(
        step0: hd_api.DiffusionStep[Batch],
    ) -> hd_api.DiffusionStep[Batch]:
      def body(
          carry: hd_api.DiffusionStep[Batch], idx: jax.Array
      ) -> tuple[hd_api.DiffusionStep[Batch], None]:
        next_info = hd_api.StepInfo(step=idx, time=ts[idx], rng=rngs[idx])
        next_step = stepper.update(
            {'x0': jnp.zeros_like(carry.xt)}, carry, next_info
        )
        return next_step, None

      final_step, _ = jax.lax.scan(body, step0, jnp.arange(1, 4))
      return final_step

    out = scan_steps(init_step)
    self.assertEqual(out.xt.shape, (2, 3))


if __name__ == '__main__':
  absltest.main()
