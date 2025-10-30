# Hackable Diffusion

Welcome to the documentation for Hackable Diffusion, a modular toolbox for
diffusion modeling in JAX.

[TOC]

## Philosophy

The core philosophy of this library is **hackability**. It is designed from the
ground up to be modular, composable, and easy to modify, enabling rapid
experimentation with new research ideas.

This is achieved through several key principles:

  * **Composition over Configuration**: Instead of large configuration files,
    the library encourages building models and training loops by composing
    small, well-defined Python objects. For example, a sampler is constructed by
    combining a `TimeSchedule`, a `SamplerStep` algorithm, and an `InferenceFn`.
  * **Clear Separation of Concerns**: The codebase is organized into logical
    sub-libraries, each responsible for a distinct part of the diffusion model
    pipeline.

## Core Design Pattern: Multimodality via "Nested" Components

A foundational concept in Hackable Diffusion is native support for
**multimodality**. The library is designed to handle complex data structures,
where different parts of your data (e.g., an image and its corresponding class
label, or a video and its audio track) can be processed with different diffusion
parameters.

This is achieved through a consistent "Nested" pattern. Most core components
have a `Nested` wrapper:

  * `NestedProcess` (`lib/corruption/base.py`)
  * `NestedDiffusionLoss` (`lib/loss/base.py`)
  * `NestedTimeSchedule` (`lib/sampling/time_scheduling.py`)
  * `NestedSamplerStep` (`lib/sampling/base.py`)

These wrappers take a PyTree (e.g., a dictionary) of component instances that
matches the structure of your data. When a method is called on the `Nested`
object, it automatically maps the corresponding function to each component-data
pair.

For example, you can define a `NestedProcess` that applies a `GaussianProcess`
to your images and a `CategoricalProcess` to your labels within the same
training step, making multimodal diffusion models first-class citizens.

## Codebase Structure

The core logic of the library resides in `lib/`. The subfolders are organized
around the key concepts of diffusion models.

### [Architecture](./architecture.md)

(`lib/architecture/`)

This module contains the building blocks for neural networks. It provides
flexible backbones like a conditional `Unet` and `MLP`, as well as a powerful
system for encoding and injecting conditioning signals via
`ConditioningEncoder`.

### [Corruption Processes](./corruption.md)

(`lib/corruption/`)

This module defines the **forward process** of diffusion. It includes
implementations for corrupting data with noise, such as `GaussianProcess` for
continuous data and `CategoricalProcess` for discrete data. It also defines the
noise `schedules` that govern the corruption over time.

### [Inference Function](./inference.md)

(`lib/inference/`)

The "Inference Function" is the core abstraction for a single denoising step. It
encapsulates the call to the model and can be composed with other
functionalities like classifier-free guidance. This allows the main sampling
loop to be agnostic to the details of how a prediction is made.

### [Diffusion Loss Functions](./loss.md)

(`lib/loss/`)

This module provides flexible loss functions for training diffusion models. It
includes highly configurable weighted MSE losses for Gaussian processes (like
`SiD2Loss`) and cross-entropy losses for discrete data.

### [Sampling](./sampling.md)

(`lib/sampling/`)

This module handles the **reverse process**—generating new data. It provides a
generic sampling loop (`DiffusionSampler`) that orchestrates a `TimeSchedule`
and a `SamplerStep` algorithm (e.g., `DDIMStep`, `SdeStep`) to iteratively
denoise a sample.

## Tutorials and Notebooks

The `notebooks/` directory contains a set of tutorials that demonstrate how to
use the library to train and sample from diffusion models. These serve as
excellent starting points for understanding the library's components in action.

  * **`2d_training.ipynb`**: A minimal example that trains a diffusion model on
    a simple 2D toy dataset.
  * **`mnist.ipynb`**: Trains a standard continuous diffusion model (Gaussian
    process) on the MNIST dataset, demonstrating image data handling.
  * **`mnist_discrete.ipynb`**: Trains a discrete diffusion model on MNIST,
    treating pixel values as categorical data. This showcases the use of
    `CategoricalProcess`.
  * **`mnist_multimodal.ipynb`**: A more advanced example that trains a
    multimodal model to jointly generate MNIST images with discrete and
    continuous diffusion models, demonstrating the "Nested" design pattern in a
    practical setting.
