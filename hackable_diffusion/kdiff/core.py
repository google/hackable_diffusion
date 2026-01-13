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

"""Core abstractions to connect hackable_diffusion and Kauldron.

Kauldron makes heavy use of the `kontext` module to dynamically pass information
to the model, losses, metrics and summaries.
The core idea is that during the execution there is a `context` object, which
contains all the currently available values (batch, params, model outputs etc.).

The model/loss/metric/summary objects have fields annotated with
`kd.kontext.Key`, which is basically a string path into the context.
When the  model/loss/metric/summary is called, kauldron uses these keys to get
the corresponding arrays from the context and passes them as arguments.

So for example the Diffusion class below has `x0` and `cond` annotated with
`kd.kontext.Key`. In the config these will be set to e.g. `batch.image` and
`batch.label`. During the execution of the model, kauldron will use these paths
to get the corresponding arrays from the context and pass them to the model.
So for example a model like this:

```python
cfg.model = core.Diffusion(
    x0="batch.image",
    cond={"label": "batch.label"},
    ...
)

So in this example the model would, during training, be effectively called with:

```python
context.preds = model(
    x0=context.batch["image"],
    cond={"label": context.batch["label"]})
```
"""

import dataclasses
from typing import Optional
import flax.linen as nn
from hackable_diffusion import hd
from hackable_diffusion.lib import hd_typing
from kauldron import kd

################################################################################
# MARK: Type aliases
################################################################################

Array = hd_typing.Array
Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
LossOutput = hd_typing.LossOutput
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray
typechecked = hd_typing.typechecked


################################################################################
# MARK: Diffusion
################################################################################


class Diffusion(nn.Module, kw_only=True):
  """Basic setup for training a diffusion model.

  This is the main entry point for training a diffusion model with
  hackable_diffusion in kauldron. It takes care of 1) Sampling timesteps t 2)
  corrupt the input data (x0) according to the corruption process to get xt 3)
  running the diffusion network to get the predictions 4) converting the network
  output to all supported prediction types (for loss)

  Attributes:
    x0: Key into the batch for the input data. E.g. 'batch.image'.
    cond: (Optional) Key tree of keys into the batch for the conditioning. E.g.
      {'label': 'batch.label'}. The tree will be passed to the __call__ function
      as `cond` argument, with the keys replaced by the corresponding values
      from the context.
    network: The diffusion network to use. This model needs to accept the
      following arguments (time, xt, cond, is_training), and it should return a
      dictionary of predictions of the form `{'prediction_type': prediction,
      ...}`
    corruption_process: The (forward) corruption process to use for corrupting
      the input data.
    time_sampler: The time sampler to use for sampling timesteps. In the
      simplest case this can just be a
      `UniformTimeSampler(safety_epsilon=1e-4)`.
  """

  network: nn.Module
  corruption_process: hd.corruption.CorruptionProcess
  time_sampler: hd.time_sampling.TimeSampler

  x0: kd.kontext.Key = kd.kontext.REQUIRED  # E.g. 'batch.image'.
  cond: Optional[kd.kontext.KeyTree] = None  # e.g. 'batch.label'
  is_training = kd.nn.train_property()

  @nn.compact
  @typechecked
  def __call__(
      self,
      x0: DataTree,
      cond: Conditioning | None = None,
  ) -> dict[str, dict[str, Array] | Array]:
    """Run the diffusion training step.

    Samples timesteps, corrupts the input data according to the corruption
    process, runs the diffusion network, computes all the predictions and
    returns a dictionary of useful outputs.

    Args:
      x0: The input data.
      cond: The conditioning (optional).

    Returns:
      A dictionary of outputs that are useful for training.
      - output: The outputs of the diffusion network with all prediction types
        filled in. So outputs will have keys like `eps`, `x0`, `velocity`, etc.
      - target: The training targets for the diffusion process. Will have the
        same keys as `output`.
      - xt: The corrupted input data.
      - noise_info: Information about the noise (sampled time, sigma, etc.)
    """

    # Sample timesteps.
    time = self.time_sampler(self.make_rng("default"), data_spec=x0)

    # Corrupt the input data according to the timesteps.
    xt, target_info = self.corruption_process.corrupt(
        self.make_rng("default"), x0, time
    )

    # Run the diffusion network
    output = self.network(
        time=time,
        xt=xt,
        conditioning=cond,
        is_training=self.is_training,
    )
    # Compute all the prediction types so they can be used in the loss/metrics.
    outputs = self.corruption_process.convert_predictions(
        prediction=output, xt=xt, time=time
    )
    # Get the noise info for summaries.
    noise_info = self.corruption_process.get_schedule_info(time)
    # Compile a dictionary of all the outputs that may be useful.
    return {
        "output": outputs,
        "target": target_info,
        "xt": xt,
        "noise_info": noise_info,
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class KauldronLossWrapper(kd.losses.Loss):
  """Wrapper for hackable diffusion loss functions."""

  # Basically just adds the kontext keys so that kauldron can pass the correct
  # predictions, targets and time arrays to the loss function.
  # These default values should work for most usecases with `Diffusion` above.
  preds: kd.kontext.Key = "preds.output"
  targets: kd.kontext.Key = "preds.target"
  time: kd.kontext.Key = "preds.noise_info.time"

  # Implicitly supports `weight` and `mask` as well (see `kd.losses.Loss`).

  loss: hd.loss.DiffusionLoss

  @typechecked
  def get_values(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    return self.loss(
        preds=preds,
        targets=targets,
        time=time,
    )
