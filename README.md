# Hackable diffusion

Hackable Diffusion is a modular toolbox written in Jax to experiment and educate
around Diffusion modeling.

## Supported Integrations

*   **Gemma Fine-Tuning**: Support for text diffusion models (e.g., DiffusionGemma) via a hybrid AR-diffusion implementation. The adapter package and fine-tuning configurations are located in the Gemma repository (GitHub: https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter).

## Philosophy

The core philosophy of this library is **hackability**. It is designed from the
ground up to be modular, composable, and easy to modify, enabling rapid
experimentation with new research ideas. Key principles include:

*   **Composition over Configuration**: Build models and training loops by composing small, well-defined Python objects.
*   **Clear Separation of Concerns**: The codebase is organized into logical sub-libraries for architecture, corruption, inference, loss, and sampling.
*   **Native Multimodality**: The library has first-class support for handling multimodal data (e.g., images and text) through a consistent "Nested" component pattern that applies different diffusion parameters to different parts of the data.

## Tutorials

The `notebooks/` directory contains several tutorials to get you started:

*   **`2d_training.ipynb`**: A minimal example on a 2D toy dataset.
*   **`mnist.ipynb`**: Standard image diffusion on MNIST.
*   **`mnist_discrete.ipynb`**: An example of discrete diffusion.
*   **`mnist_multimodal.ipynb`**: A showcase of the multimodal capabilities, generating images and labels jointly.

## Training configs

The `kdiff/configs/` directory contains example configurations for training:

*   **`mnist_unet.py`**: Standard diffusion training configuration on MNIST.

To run a config locally, create a small launcher script (e.g. `train.py`):

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import multiprocessing
from kauldron import konfig

def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config", "kdiff/configs/mnist_unet.py"
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    cfg = config_module.get_config()
    cfg.workdir = "/tmp/mnist_workdir"
    trainer = konfig.resolve(cfg)
    trainer.train()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
```

> **Note:** `XLA_PYTHON_CLIENT_PREALLOCATE=false` must be set *before*
> importing JAX to prevent GPU memory preallocation conflicts with data
> loading workers. The `if __name__ == "__main__"` guard is required for
> multiprocessing compatibility.

## Installation

To install the necessary dependencies, you can use pip with the provided
`pyproject.toml` file:

```bash
pip install -e .
```

To install development dependencies (for running tests), use:

```bash
pip install -e .[dev]
```

This will install libraries such as JAX, Flax, and other utilities required to
run the code.


## Disclaimer

Copyright 2025 Google LLC \
All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0 All other materials are licensed
under the Creative Commons Attribution 4.0 International License (CC-BY). You
may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode Unless required by
applicable law or agreed to in writing, all software and materials distributed
here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
licenses for the specific language governing permissions and limitations under
those licenses.
This is not an official Google product.


## Citing Hackable Diffusion

If Hackable Diffusion was helpful for a publication, please cite this repository:
(authors are included in the alphabetical order by the last name)

```
@software{hackable_diffusion2026github,
  author = {Crepy, Clement and De Bortoli, Valentin and Galashov, Alexandre and Greff, Klaus and Korshunova, Ira},
  title = {{Hackable Diffusion}: A modular toolbox written in Jax to experiment and educate around Diffusion modeling.},
  url = {https://github.com/google/hackable_diffusion},
  version = {1.0.1},
  year = {2026},
  note = {Authors listed in alphabetical order by the last name},
}
```

*This is not an officially supported Google product.*