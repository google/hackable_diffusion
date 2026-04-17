# Corruption Processes

This document describes the corruption processes used in the Hackable Diffusion
library, which correspond to the forward process in diffusion models. These
modules are responsible for adding noise to the data.

The modules related to corruption are located in `lib/corruption/`.

[TOC]

## Overview

The "corruption" process defines how clean data `x0` is gradually transformed
into noise. This library provides a flexible framework for defining and using
various corruption processes for both continuous and discrete data.

The main components are:

*   **`CorruptionProcess` Protocol**: An interface that standardizes how
    corruption is applied.
*   **Schedules**: Functions that define the rate and nature of corruption over
    time `t`.
*   **Process Implementations**: Concrete classes like `GaussianProcess` for
    continuous data (e.g., images), `CategoricalProcess` for discrete data
    (e.g., labels, tokens), `SimplicialProcess` for simplex-valued categorical
    data (e.g., graph edges with Dirichlet noise), and `RiemannianProcess` for
    data on Riemannian manifolds..

## `CorruptionProcess` Protocol

(`lib/corruption/base.py`)

The `CorruptionProcess` is a protocol (an interface) that all corruption
processes must implement. It defines the following key methods:

*   `corrupt(key, x0, time)`: Takes clean data `x0` and a time `t`, and returns
    the corrupted data `xt` along with a dictionary of potential training
    targets and metadata (`target_info`).
*   `sample_from_invariant(key, data_spec)`: Samples from the invariant
    distribution of the process (i.e., pure noise at `t=1`).
*   `convert_predictions(prediction, xt, time)`: Takes a model's prediction
    (e.g., predicted epsilon) and converts it into all other possible target
    parameterizations (e.g., predicted `x0`, score, etc.).
*   `get_schedule_info(time)`: Returns parameters of the schedule at a given
    time.

### `NestedProcess`

(`lib/multimodal.py`)

For handling complex data structures (pytrees), `NestedProcess` is a wrapper
that applies different corruption processes to different leaves of the pytree.
For example, you can use a `GaussianProcess` on an image and a
`CategoricalProcess` on its corresponding labels simultaneously.

## Schedules

(`lib/corruption/schedules.py`)

Schedules define how the corruption parameters change over the continuous time
interval `[0, 1]`.

### `GaussianSchedule`

For Gaussian processes, schedules define `alpha(t)` and `sigma(t)`. Common
implementations include:

*   `CosineSchedule`: A popular choice where `alpha(t) = cos(0.5 * pi * t)`.
*   `RFSchedule`: Rectified Flow schedule where `alpha(t) = 1 - t` and
    `sigma(t) = t`.
*   `LinearDiffusionSchedule`: The schedule from the original DDPM paper,
    parameterized by `beta_min` and `beta_max`.

### `DiscreteSchedule` / `SimplicialSchedule`

For discrete and simplicial processes, schedules define `alpha(t)`, which
controls the signal-to-noise ratio at time `t`. `SimplicialSchedule` is an alias
for `DiscreteSchedule`; they share the same implementations:

*   `LinearDiscreteSchedule`: `alpha(t) = 1 - t`.
*   `CosineDiscreteSchedule`: `alpha(t) = cos(0.5 * pi * t)`.

For discrete processes, `alpha(t)` is interpreted as the probability of
*keeping* the original token. For simplicial processes, it parameterises the
signal-to-noise ratio via `h(t) = alpha(t) / (1 - alpha(t))`.

## `GaussianProcess`

(`lib/corruption/gaussian.py`)

This is the implementation for standard diffusion on continuous data. It defines
the corruption as:

`xt = alpha(t) * x0 + sigma(t) * epsilon`

where `epsilon` is standard Gaussian noise with unit variance.

### Prediction Parameterizations

A key feature of `GaussianProcess` is its handling of different prediction
targets. The denoising model can be trained to predict various quantities. The
`corrupt` method returns all of them in `target_info`, and `convert_predictions`
can switch between them.

The supported parameterizations are:

*   **`x0`**: Predict the original clean data.
*   **`epsilon`**: Predict the noise that was added.
*   **`score`**: Predict the score function (gradient of the log-density), which
    is `-epsilon / sigma(t)`.
*   **`velocity`**: The velocity field using in Flow Matching
    (<https://arxiv.org/abs/2210.02747>), Rectified Flow
    (<https://arxiv.org/abs/2209.03003>) and Stochastic Interpolants
    (<https://arxiv.org/abs/2303.08797>) implementations.
*   **`v`**: The `v-prediction` first introduced in Progressive Distillation
    (<https://arxiv.org/abs/2202.00512>).

This flexibility allows you to train a model with one objective (e.g.,
epsilon-prediction) but use a sampler that requires a different one (e.g.,
x0-prediction) without any code change in the sampler.

### Example Usage

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.schedules import CosineSchedule

key = jax.random.PRNGKey(0)

# 1. Define schedule and process
schedule = CosineSchedule()
process = GaussianProcess(schedule=schedule)

# 2. Create data and time
x0 = jnp.ones((1, 32, 32, 3))
time = jnp.array([0.5]) # Corrupt halfway

# 3. Apply corruption
key, subkey = jax.random.split(key)
xt, target_info = process.corrupt(subkey, x0, time)

print(f"Shape of xt: {xt.shape}")
print(f"Available targets: {target_info.keys()}")

# 4. Convert between predictions
# Suppose a model predicts epsilon
model_prediction = {'epsilon': jnp.zeros_like(x0)}

# Get all other parameterizations
all_predictions = process.convert_predictions(model_prediction, xt, time)
print(f"Converted predictions: {all_predictions.keys()}")

# You can access the predicted x0
predicted_x0 = all_predictions['x0']
```

## `CategoricalProcess`

(`lib/corruption/discrete.py`)

This process is designed for discrete data, such as integer class labels or
tokens. At each time `t`, it replaces original tokens with noise tokens with
probability `1 - alpha(t)`.

Key configuration parameters:

*   `schedule`: A `DiscreteSchedule` that defines `alpha(t)`.
*   `invariant_probs`: The probability distribution of the "noise" tokens that
    replace the original ones.
*   `num_categories`: The number of valid categories in the data.
*   `unused_token`: This corresponds to a value of a token which should not be
    modelled by the diffusion process and which is not part of the vocabulary.
    Note that in the text diffusion, this is not typically a padding token,
    because it is a part of vocabulary and is modelled by diffusion. An example
    of such a token is in graph diffusion of adjacency matrix where this token
    can essentially put some structure on this matrix and forbid some dimensions
    to be unmasked. It can be interpreted as a form of padding in this context.

The library provides convenient factory methods for common use cases:

*   `CategoricalProcess.uniform_process`: Corrupts tokens by replacing them with
    a token drawn uniformly from all possible categories.
*   `CategoricalProcess.masking_process`: Corrupts tokens by replacing them with
    a special "mask" token. This requires `num_categories` to be the vocabulary
    size, and the mask token will be integer `num_categories`.

The `corrupt` function returns `target_info` which contains `x0`, `logits`, as
well as different masks such as `is_corrupted` and `is_unused`. The mask
`is_corrupted` is true for all tokens which are corrupted and not unused (a
token which is `unused_token` has mask equal to False). The mask `is_unused` is
only true for tokens which value is equal to `unused_token`. The mask
`is_corrupted` is used in the loss function to compute it only on the corrupted
tokens. The mask `is_unused` could be used in case if the loss needs to be
computed on all but unused tokens.

### Example Usage (Masking)

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.corruption.discrete import CategoricalProcess
from hackable_diffusion.lib.corruption.schedules import LinearDiscreteSchedule

key = jax.random.PRNGKey(0)
num_classes = 10 #
mask_token_id = num_classes

# 1. Create a masking process
schedule = LinearDiscreteSchedule()
process = CategoricalProcess.masking_process(
    schedule=schedule,
    num_categories=num_classes,
)

# 2. Data and time
# Shape is (batch, sequence_length, 1)
x0 = jnp.array([1, 2, 3, 4]).reshape(1, 4, 1)
time = jnp.array([0.8]) # Corrupt heavily, most tokens should be masked

# 3. Corrupt data
key, subkey = jax.random.split(key)
xt, target_info = process.corrupt(subkey, x0, time)

print(f"Original x0: {x0.flatten()}")
print(f"Corrupted xt: {xt.flatten()}")
# With high probability, most tokens in xt will be `mask_token_id` (10)

# The target info contains the one-hot encoded version of x0
print(f"Logits target shape: {target_info['logits'].shape}")
# Logits target shape: (1, 4, 10)
```

**Assumptions**:

*   Discrete data is expected to be integer arrays with a trailing dimension
    of 1.
*   The model prediction for discrete data is expected to be logits over the
    categories. `convert_predictions` will then convert these logits to a
    predicted `x0` (via argmax).

## `RiemannianProcess`

(`lib/corruption/riemannian.py`)

This process implements **Riemannian Flow Matching (RFM)**, a generalization of
Flow Matching to smooth Riemannian manifolds. Unlike standard diffusion, which
relies on Gaussian noise, RFM uses the manifold's intrinsic geometry to
interpolate between data and noise distributions.

### Mathematical Foundations: Continuous-time Flow Matching

Let $$(M, g)$$ be a $$d$$-dimensional smooth Riemannian manifold. A probability
path $$p_t$$ on $$M$$ can be defined via the **Continuity Equation**:

$$\frac{\partial p_t}{\partial t} + \operatorname{div}_g (p_t v_t) = 0$$

where $$\operatorname{div}_g$$ is the Riemannian divergence operator and $$v_t
\in T_x M$$ is a time-dependent vector field. Riemannian Flow Matching aims to
find a vector field $$v_\theta(x, t)$$ that generates a path $$p_t$$ such that
$$p_0$$ is the data distribution and $$p_1$$ is an invariant noise distribution.

### Riemannian Concepts: Exp, Log, and Geodesics

The geometry of the manifold is abstracted through three key operations
implemented in `lib/manifolds.py`:

#### 1. Exponential Mapping ($$\text{Exp}_x$$)

The exponential map $$\text{Exp}_x : T_x M \to M$$ provides a way to "map" a
tangent vector $$v \in T_x M$$ back onto the manifold. Intuitively, if you start
at point $$x$$ and walk in the direction of $$v$$ for unit time along the unique
"straightest" path (geodesic), you arrive at $$\text{Exp}_x(v)$$.

In the library, this is used during **sampling** (to move from $$x_t$$ to
$$x_{t-dt}$$) and to construct geodesics.

#### 2. Logarithm Mapping ($$\text{Log}_x$$)

The logarithm map $$\text{Log}_x : M \to T_x M$$ is the inverse of the
exponential map (where defined). Given two points $$x, y \in M$$,
$$\text{Log}_x(y)$$ returns the tangent vector at $$x$$ that points toward $$y$$
along the shortest geodesic. The length of this vector equals the Riemannian
distance between the two points: $$\|\text{Log}_x(y)\|_g = d_g(x, y)$$.

In the library, this is used during **training** to find the direction of the
conditional flow between noise and data.

#### 3. Geodesic Mapping ($$\gamma$$)

A geodesic is the generalization of a straight line to curved spaces. The unique
geodesic path starting at $$x$$ and ending at $$y$$ can be parameterized by $$t
\in [0, 1]$$ as:

$$\gamma(t) = \text{Exp}_x(t \cdot \text{Log}_x(y))$$

This mapping ensures that the interpolation between distributions stays on the
manifold and follows the shortest possible paths, which is the cornerstone of
Riemannian Flow Matching.

### The Riemannian Flow Matching loss

$$L(\theta) = \mathbb{E}_{t \sim U[0, 1], x_0 \sim p_0, x_1 \sim p_1} [ \| v_{\theta}(x_t, t) - u_t(x_t | x_0, x_1) \|_{g}^2 ]$$

where the conditional velocity field $$u_t(x|x_0, x_1)$$ is derived from a
conditional probability path $$p_t(x|x_0, x_1)$$ that satisfies the continuity
equation. In this library, we use **geodesic paths** for the conditional
interpolation:

1.  **Conditional Path**: $$x_t = \text{Exp}_{x_1}(\alpha(t)
    \text{Log}_{x_1}(x_0))$$
2.  **Conditional Velocity**: $$u_t(x_t | x_0, x_1) = \dot{\alpha}(t) \cdot
    \frac{d}{ds} \text{Exp}_{x_1}(s \text{Log}_{x_1}(x_0)) \big|_{s=\alpha(t)}$$

For the standard `LinearRiemannianSchedule`, $$\alpha(t) = 1 - t$$, meaning the
path flows from noise ($$t=0, \alpha=1, x_{t=0}=x_1$$) to data ($$t=1, \alpha=0,
x_{t=1}=x_0$$). *Note: The implementation uses $$\alpha(t)$$ such that $$t=0$$
is clean data and $$t=1$$ is noise, with internal interpolation adjustments to
match this theory.*

### Supported Manifolds (`lib/manifolds.py`)

Each manifold implements the `Manifold` protocol, providing core geometric
operations with an emphasis on numerical stability.

#### 1. Unit Hypersphere ($$S^d$$)

Points $$x \in \mathbb{R}^{d+1}$$ such that $$\|x\|_2 = 1$$. The tangent space
$$T_x S^d$$ is the subspace $$\{v \in \mathbb{R}^{d+1} \mid \langle x, v
\rangle = 0\}$$.

*   **Exp**: $$\text{Exp}_x(v) = \cos(\|v\|)x + \text{sinc}(\|v\|)v$$
*   **Log**: $$\text{Log}_x(y) = \frac{\theta}{\sin \theta}(y - \cos \theta
    x)$$, where $$\theta = \arccos(\langle x, y \rangle)$$
*   **Velocity**: The time-derivative along the geodesic: $$u_t = -\theta
    \sin(\theta t)x_1 + \cos(\theta t) \text{Log}_{x_1}(x_0)$$

The implementation uses an **unnormalized sinc trick** ($$\text{sinc}(x) =
\frac{\sin x}{x}$$) to handle the singularity at $$\theta=0$$ gracefully.

#### 2. Special Orthogonal Group ($$SO(3)$$)

Points $$R$$ are $$3 \times 3$$ rotation matrices. The tangent space $$T_R
SO(3)$$ is isomorphic to the Lie Algebra $$\mathfrak{so}(3)$$ of skew-symmetric
matrices via $$R \cdot \omega^\wedge$$.

*   **Exp**: Computed via **Rodrigues' Rotation Formula**: $$\text{Exp}_R(v) = R
    (I + \text{sinc}(\theta)\omega^\wedge +
    \text{cosc}(\theta)(\omega^\wedge)^2)$$, where $$\theta = \|\omega\|$$.
*   **Log**: Maps $$R_1^T R_0$$ to its rotation axis and angle $$\theta$$.
*   **Velocity**: $$u_t = x_t \cdot \text{Log}(x_1^T x_0)$$.

The library uses a safe **cosc trick** ($$\text{cosc}(x) = \frac{1 - \cos
x}{x^2} = \frac{1}{2} \text{sinc}(\frac{x}{2})^2$$) to ensure numerical
stability in the Rodrigues formula.

#### 3. Flat Torus ($[0, 1]^d$)

The torus is a flat space with periodic boundary conditions.

*   **Metric**: Standard Euclidean metric $$g = I$$.
*   **Geodesics**: Straight lines modulo 1.
*   **Velocity**: Constant velocity $$u = \text{Log}_{x_1}(x_0) = (x_0 - x_1 +
    0.5) \pmod 1 - 0.5$$.

### Example Usage

```python
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption.riemannian import RiemannianProcess
from hackable_diffusion.lib.corruption.schedules import LinearRiemannianSchedule

# 1. Define manifold and process
manifold = manifolds.Sphere()
schedule = LinearRiemannianSchedule()
process = RiemannianProcess(manifold=manifold, schedule=schedule)

# 2. Corrupt data
x0 = jnp.array([[1.0, 0.0, 0.0]]) # Point on S2
time = jnp.array([0.5])
xt, target_info = process.corrupt(subkey, x0, time)

# target_info['velocity'] is the regression target u_t
```

## `SimplicialProcess`

(`lib/corruption/simplicial.py`)

The simplicial process is a corruption process for **categorical data on the
probability simplex**. Instead of representing categorical states as integer
tokens (as in `CategoricalProcess`), each data point is represented as a
continuous probability vector on the (K-1)-simplex, and corruption adds
Dirichlet noise.

This has a key advantage called **intrinsic self-conditioning**: the corrupted
state `P_t` is itself a probability distribution over categories. The model
receives this soft distribution as input, giving it access to an implicit signal
about the prediction — without requiring an explicit self-conditioning
mechanism.

### Mathematical framework

The process is parameterised by three quantities:

*   **Invariant distribution** `pi`: a probability vector of length K (or K+1
    for masking). This is the target distribution at `t=1`.
*   **Temperature** `epsilon` (`eps`): a positive scalar controlling the
    concentration of the Dirichlet distribution — larger `eps` means softer
    (more diffuse) distributions.
*   **Schedule** `alpha(t)`: a monotonically decreasing function from
    `alpha(0) = 1` to `alpha(1) = 0`.

The **h-function** converts the schedule into a signal-to-noise ratio:

```
h(t) = alpha(t) / (1 - alpha(t))
```

At `t=0`, `h(0) = +inf` (clean data). At `t=1`, `h(1) = 0` (pure noise).

The **forward corruption** at time `t` given clean data `x0` is:

```
P_t ~ Dir(eps * (pi + h(t) * delta(x0)))
```

where `delta(x0)` is the one-hot encoding of `x0`. This Dirichlet distribution
concentrates on `delta(x0)` when `h(t)` is large (near `t=0`) and on `pi` when
`h(t)` is small (near `t=1`).

All computations are performed in **log-space** for numerical stability: the
corrupted state `xt` stores `log(P_t)` rather than `P_t` directly. This is
achieved using fast log-Gamma samplers followed by log-softmax normalization
(see `fast_random.log_dirichlet_fast`).

### Comparison with `CategoricalProcess`

| Aspect                | `CategoricalProcess`    | `SimplicialProcess`        |
| --------------------- | ----------------------- | -------------------------- |
| Corrupted state type  | Integer token (hard)    | Probability vector on      |
:                       :                         : simplex (soft)             :
| Noise mechanism       | Replace token with prob | Sample from `Dir(eps*(pi + |
:                       : `1-alpha(t)`            : h(t)*delta(x0)))`          :
| Model input           | One-hot or embedding of | Log-probability vector     |
:                       : integer                 :                            :
| Self-conditioning     | Requires explicit       | Intrinsic (P_t is itself a |
:                       : mechanism               : distribution)              :
| Output representation | Integer array `[B, ..., | Log-prob array `[B, ...,   |
:                       : 1]`                     : K]`                        :

### Configuration

Key attributes of `SimplicialProcess`:

*   `schedule`: A `SimplicialSchedule` (alias for `DiscreteSchedule`) that
    defines `alpha(t)`.
*   `invariant_probs`: Tuple of floats defining `pi`. For uniform: `(1/K, ...,
    1/K)`. For masking: `(0, ..., 0, 1)`.
*   `num_categories`: The number of categories `K` in the data (may differ from
    `len(invariant_probs)` for masking, where the mask adds one extra category).
*   `temperature`: The Dirichlet temperature `eps` (default `1.0`).
*   `unused_token`: Integer value for tokens that should not be modelled
    (default `-1`).
*   `safety_epsilon`: Small constant added to the denominator of `h(t)` to avoid
    division by zero (default `1e-6`).
*   `post_corruption_fn`: A projection applied after each corruption or sampling
    step to enforce structural constraints (default: identity).

### Factory methods

```python
from hackable_diffusion.lib.corruption.simplicial import SimplicialProcess
from hackable_diffusion.lib.corruption.schedules import CosineDiscreteSchedule

schedule = CosineDiscreteSchedule()

# Uniform invariant distribution: pi = (1/K, ..., 1/K)
process = SimplicialProcess.uniform_process(
    schedule=schedule,
    num_categories=5,
    temperature=1.0,
)

# Masking invariant distribution: pi = (0, ..., 0, 1)
process = SimplicialProcess.masking_process(
    schedule=schedule,
    num_categories=5,
    temperature=1.0,
)
```

### Example Usage

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.corruption.simplicial import SimplicialProcess
from hackable_diffusion.lib.corruption.schedules import CosineDiscreteSchedule

key = jax.random.PRNGKey(0)
num_categories = 5

# 1. Create a uniform simplicial process
schedule = CosineDiscreteSchedule()
process = SimplicialProcess.uniform_process(
    schedule=schedule,
    num_categories=num_categories,
    temperature=1.0,
)

# 2. Data: integer tokens with trailing dim of 1
x0 = jnp.array([0, 1, 2, 3]).reshape(1, 4, 1)
time = jnp.array([0.5])

# 3. Corrupt: xt is a log-probability array of shape (1, 4, 5)
key, subkey = jax.random.split(key)
xt, target_info = process.corrupt(subkey, x0, time)

print(f"xt shape: {xt.shape}")       # (1, 4, 5)
print(f"xt is log-probs: {jnp.exp(xt).sum(axis=-1)}")  # ≈ 1.0

# 4. Sample from the invariant (pure noise)
key, subkey = jax.random.split(key)
noise = process.sample_from_invariant(subkey, x0)
print(f"noise shape: {noise.shape}")  # (1, 4, 5)
```

### Post-Corruption Functions

(`lib/corruption/simplicial.py`)

Post-corruption functions are projections applied to the log-probability array
after each forward-corruption step and after each reverse-diffusion step. They
enforce structural constraints on the noisy state.

The protocol is `SimplicialPostCorruptionFn`, which takes and returns a
log-probability array.

Implementations:

*   **`IdentitySimplicialPostCorruptionFn`**: No-op (default). Returns the input
    unchanged.

*   **`SymmetricSimplicialPostCorruptionFn`**: For graph diffusion, enforces
    that edge `(i, j)` and edge `(j, i)` share the same categorical
    distribution, and zeroes out diagonal entries (no self-loops). This is the
    simplicial analogue of `SymmetricPostCorruptionFn` from the discrete
    process. Input shape must be `(batch, N, N, K)`.

    The symmetrisation uses the same `triu + transpose` pattern as the discrete
    version: extract the upper triangle, copy it to the lower triangle, and set
    diagonal entries to a "no-edge" log-probability vector (all mass on category
    0).

```python
from hackable_diffusion.lib.corruption.simplicial import (
    SimplicialProcess,
    SymmetricSimplicialPostCorruptionFn,
)
from hackable_diffusion.lib.corruption.schedules import CosineDiscreteSchedule

# For graph diffusion with symmetric adjacency matrices
process = SimplicialProcess.uniform_process(
    schedule=CosineDiscreteSchedule(),
    num_categories=3,  # e.g., no-edge, single, double bond
    post_corruption_fn=SymmetricSimplicialPostCorruptionFn(),
)
```
