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

r"""Simple diffusion config for training a UNet on MNIST.
"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import optax
  from jax.experimental import checkify
  from hackable_diffusion import hd
  from hackable_diffusion.kdiff import core
  from hackable_diffusion.kdiff import evals


# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg._konfig_experimental_nofreeze = True  # pylint: disable=protected-access
  cfg.seed = 1337

  cfg.aux = konfig.ConfigDict()
  cfg.aux.cond_embedding_dim = 128

  ##############################################################################
  # MARK: Corruption Process
  ##############################################################################

  corruption_process = hd.corruption.GaussianProcess(
      schedule=hd.corruption.RFSchedule(),
  )

  ##############################################################################
  # MARK: Conditioning networks
  ##############################################################################

  time_encoder = hd.architecture.SinusoidalTimeEmbedder(
      activation="gelu",
      num_features=cfg.ref.aux.cond_embedding_dim,
      embedding_dim=cfg.ref.aux.cond_embedding_dim,
  )
  label_encoder = hd.architecture.LabelEmbedder(
      num_classes=10,
      num_features=cfg.ref.aux.cond_embedding_dim,
      conditioning_key="label",
  )
  conditioning_encoder = hd.architecture.ConditioningEncoder(
      time_embedder=time_encoder,
      conditioning_embedders={
          "label": label_encoder,
      },
      merge_embeddings_fn=hd.architecture.SumEmbeddings(),
      conditioning_rules={
          "label": "adaptive_norm",
          "time": "adaptive_norm",
      },
      conditioning_dropout_rate=0.1,
  )

  ##############################################################################
  # MARK: Backbone network
  ##############################################################################

  backbone_network = hd.architecture.Unet(
      base_channels=64,
      channels_multiplier=(1, 2, 2, 2),
      num_residual_blocks=(2, 2, 2, 2),
      downsample_fn=hd.architecture.AvgPoolDownsample(),
      upsample_fn=hd.architecture.ImageResizeUpsample(resize_method="nearest"),
      dropout_rate=(0.0, 0.1, 0.1, 0.1),
      bottleneck_dropout_rate=0.1,
      self_attention_bool=(False, False, True, True),
      cross_attention_bool=(False, False, False, False),
      attention_normalize_qk=True,
      attention_use_rope=True,
      attention_rope_positions_fn=hd.architecture.SquareRoPEPositions(),
      attention_num_heads=-1,
      attention_head_dim=64,
      uncond_norm_strategy=hd.architecture.RMSNormStrategy(),
      cond_norm_strategy=hd.architecture.ConditionalRMSNormStrategy(
          use_shift=True
      ),
      activation="gelu",
      skip_connection_fn=hd.architecture.UnnormalizedAddSkip(),
  )

  ##############################################################################
  # MARK: Diffusion model
  ##############################################################################

  cfg.model = core.Diffusion(
      x0="batch.image",
      cond={"label": "batch.label[:,0]"},
      corruption_process=corruption_process,
      time_sampler=hd.training.time_sampling.UniformTimeSampler(
          span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4)
      ),
      network=hd.diffusion_network.DiffusionNetwork(
          prediction_type="velocity",
          backbone_network=backbone_network,
          conditioning_encoder=conditioning_encoder,
      ),
  )

  ##############################################################################
  # MARK: Training
  ##############################################################################

  cfg.num_train_steps = 100_000

  cfg.train_ds = _make_ds(training=True, batch_size=256)

  ##############################################################################
  # MARK: Losses
  ##############################################################################

  cfg.train_losses = {
      "diffusion_loss": core.KauldronLossWrapper(
          loss=hd.training.SiD2Loss(
              schedule=cfg.ref.model.corruption_process.schedule,
              prediction_type=cfg.ref.model.network.prediction_type,
              bias=2.0,
          ),
      ),
  }

  ##############################################################################
  # MARK: Optimizer
  ##############################################################################

  cfg.schedules = {
      "learning_rate": optax.warmup_constant_schedule(
          init_value=0.0,
          peak_value=3e-4,
          warmup_steps=1_000,
      )
  }

  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adam": optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-12),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
      "ema": kd.optim.ema_params(decay=0.999),
  })

  #############################################################################
  # MARK: Metrics
  ##############################################################################

  cfg.train_metrics = {
      "grad_norm": kd.metrics.SkipIfMissing(
          kd.metrics.TreeReduce(
              metric=kd.metrics.Norm(
                  tensor="grads", axis=None, aggregation_type="concat"
              )
          )
      ),
  }
  cfg.train_summaries = {
      "gt": kd.summaries.ShowImages(
          images="batch.image", in_vrange=(-1.0, 1.0)
      ),
      "x0_pred": kd.summaries.ShowImages(
          images="preds.output.x0", in_vrange=(-1.0, 1.0)
      ),
      "xt": kd.summaries.ShowImages(images="preds.xt", in_vrange=(-1.0, 1.0)),
  }

  ##############################################################################
  # MARK: Evals
  ##############################################################################

  cfg.eval_ds = _make_ds(training=False, batch_size=256)

  cfg.evals = {
      "sample_DDIM": evals.SamplingEvaluator(
          run=kd.evals.EveryNSteps(10_000, skip_first=True),
          num_batches=None,
          sampler=hd.sampling.DiffusionSampler(
              time_schedule=hd.sampling.UniformTimeSchedule(),
              stepper=hd.sampling.DDIMStep(
                  stoch_coeff=0.0,
                  corruption_process=cfg.ref.model.corruption_process,
              ),
              num_steps=50,
          ),
          metrics={},
          summaries={
              "gt": kd.summaries.ShowImages(
                  images="batch.image", in_vrange=(-1.0, 1.0)
              ),
              "sample": kd.summaries.ShowImages(
                  images="samples.xt", in_vrange=(-1.0, 1.0)
              ),
          },
      ),
  }

  ##############################################################################
  # MARK: Checkpointer
  ##############################################################################

  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=10_000,
      max_to_keep=3,
  )

  ##############################################################################
  # MARK: Other
  ##############################################################################

  # hackable diffusion requires checkify to be activated.
  cfg.checkify_error_categories = checkify.user_checks
  # Set up random streams.
  cfg.rng_streams = kd.train.RngStreams([
      # The SamplingEvaluator uses the "sampling" stream.
      kd.train.RngStream("default", train=True, eval=True),
      kd.train.RngStream("sampling", train=True, eval=True),
  ])

  return cfg


################################################################################
# MARK: Make dataset
################################################################################


def _make_ds(training: bool, batch_size: int, split: str | None = None):
  """MNIST dataset."""
  transforms = [
      kd.data.Elements(keep=["image", "label"]),
      kd.data.py.Resize(key="image", size=(32, 32), method="bilinear"),
      kd.data.ValueRange(key="image", in_vrange=(0, 255), vrange=(-1, 1)),
      kd.data.Rearrange(key="label", pattern="... -> ... 1"),
  ]
  if training:
    # No random flip for MNIST as digits are not flip-invariant
    pass

  if split is None:
    split = "train" if training else "test"

  return kd.data.py.Tfds(
      name="mnist",
      split=split,
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      transforms=transforms,
      batch_size=batch_size,
  )
