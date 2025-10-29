# Hackable diffusion

Hackable Diffusion is a modular toolbox written in Jax to experiment and educate
around Diffusion modeling.

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

## Installation

To install the necessary dependencies, you can use pip with the provided
`requirements.txt` file:

```bash
pip install -r requirements.txt
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
