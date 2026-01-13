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

"""Sampling evaluator for Diffusion models."""

import dataclasses
from typing import Any
from flax import linen as nn
import flax.struct
from hackable_diffusion import hd
from hackable_diffusion.lib import hd_typing
import jax
from kauldron import kd
import kauldron.data.utils as data_utils


################################################################################
# MARK: Type aliases
################################################################################

Array = hd_typing.Array
Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree

################################################################################
# MARK: InferenceFnFactory
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class KDiffInferenceFn:
  """Create an inference function for a KDiff diffusion model."""

  network_path: str = "network"

  def from_model_and_context(
      self, model: nn.Module, context: kd.train.Context
  ) -> hd.inference.InferenceFn:
    network = kd.kontext.get_by_path(model, self.network_path)
    params = kd.kontext.get_by_path(context.params, self.network_path)

    inference_fn = hd.inference.FlaxLinenInferenceFn(
        network=network, params=params
    )

    return inference_fn


################################################################################
# MARK: SamplingEvaluator
################################################################################


class SamplingEvaluator(kd.contrib.evals.CheckpointedEvaluator):
  """Evaluator that samples from the model.

  Attributes:
    sampler: SampleFn to use for sampling.
    inference: Inference function to use for sampling.
    rng_stream: The rng stream to use for sampling.

  Example:

    ```python
    cfg.evals = {
        "sample_DDIM": evals.SamplingEvaluator(
            run=kd.evals.EveryNSteps(50_000, skip_first=True),
            checkpointer=kd.ckpts.Checkpointer(
                save_interval_steps=5,
            ),
            sampler=hd.sampling.DiffusionSampler(
                time_schedule=hd.sampling.UniformTimeSchedule(),
                stepper=hd.sampling.DDIMStep(
                    stoch_coeff=0.25,
                    corruption_process=cfg.ref.model.corruption_process,
                ),
                num_steps=250,
            ),
            metrics={
                ...
            },
            summaries={
                ...
            },
        ),
    }
    ```
  """

  sampler: hd.sampling.SampleFn

  num_batches: int | None

  inference: KDiffInferenceFn = KDiffInferenceFn()

  rng_stream: str = "sampling"

  # override the default values for losses, metrics, and summaries to be empty
  # (because the training ones likely don't make sense for sampling)
  losses: dict[str, kd.losses.Loss] = dataclasses.field(default_factory=dict)
  metrics: dict[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )
  summaries: dict[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )

  # Set the default checkpointer to a noop checkpointer.
  checkpointer: kd.ckpts.BaseCheckpointer = kd.ckpts.NoopCheckpointer()

  def _sample_initializer(
      self,
      model: nn.Module,
      context: kd.train.Context,
      key: hd.hd_typing.PRNGKey,
  ) -> tuple[DataTree, Array["batch *cond_shape"]]:
    _, kwargs = data_utils.get_model_inputs(model, context)
    x0 = kwargs["x0"]
    cond = kwargs.get("cond", None)  # cond is optional
    x1 = model.corruption_process.sample_from_invariant(key, x0)
    return x1, cond

  def _step(
      self, step_nr: int, state: kd.train.TrainState, batch: Any
  ) -> kd.train.AuxiliariesState:
    """Custom sampling eval step."""
    # Set up the context and the inference function
    context = SamplingContext.from_state_and_batch(state=state, batch=batch)
    inference_fn = self.inference.from_model_and_context(self.model, context)

    # Create PRNG keys for init and sampling
    rngs = self.base_cfg.rng_streams.eval_rngs(step_nr)
    init_rng, sample_rng = jax.random.split(rngs[self.rng_stream], 2)

    x1, cond = self._sample_initializer(self.model, context, init_rng)

    # Run the sampling loop
    final, interm = self.sampler(
        inference_fn=inference_fn,
        rng=sample_rng,
        initial_noise=x1,
        conditioning=cond,
    )

    # Update the context with the final and intermediate samples
    # final, and interms are DiffusionStep trees.
    context = context.replace(
        samples=final,
        sample_interms=interm,
    )
    # Compute the metrics
    context = self.aux.update_context(context)
    return context.get_aux_state(
        return_losses=True, return_metrics=True, return_summaries=True
    )

  def __hash__(self) -> int:
    # Make Evaluator hashable, so its methods can be jitted.
    return id(self)


################################################################################
# MARK: SamplingContext
################################################################################


@flax.struct.dataclass
class SamplingContext(kd.train.Context):
  """Context with additional fields for sampling."""

  samples: Any = None
  sample_interms: Any = None
