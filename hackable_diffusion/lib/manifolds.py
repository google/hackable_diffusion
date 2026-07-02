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

"""Manifolds."""

from typing import Protocol
from hackable_diffusion.lib import hd_typing
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

DataArray = hd_typing.DataArray
TimeArray = hd_typing.TimeArray
PRNGKey = hd_typing.PRNGKey
LossOutput = hd_typing.LossOutput
Array = hd_typing.Array

################################################################################
# MARK: Constants
################################################################################

EPSILON = 1e-9

################################################################################
# MARK: Utility Functions
################################################################################


def unnormalized_sinc(
    x: Array['*batch'],  # pyrefly: ignore[not-a-type]
) -> Array['*batch']:  # pyrefly: ignore[not-a-type]
  """Safe sinc(x)."""
  return jnp.sinc(x / jnp.pi)


def unnormalized_cosc(
    x: Array['*batch'],  # pyrefly: ignore[not-a-type]
) -> Array['*batch']:  # pyrefly: ignore[not-a-type]
  """Safe (1-cos(x))/x^2.

  Leverages the sinc trick to compute (1-cos(x))/x^2 safely. Using the identity

    (1 - cos(x)) / x^2 = 0.5 * sinc(x / 2)^2

  Args:
    x: Input angles.

  Returns:
    Safe (1-cos(x))/x^2.
  """
  return 0.5 * unnormalized_sinc(x / 2.0) ** 2


def safe_norm(
    x: Array,  # pyrefly: ignore[not-a-type]
    axis: tuple[int, ...] = (-1,),
    keepdims: bool = True,
    eps: float = 1e-9,
) -> Array:  # pyrefly: ignore[not-a-type]
  """Computes norm safely to avoid NaN gradients at zero.

  Uses ``sqrt(max(sum(x^2), eps^2))`` instead of the standard ``norm``.

  Args:
    x: Input tensor.
    axis: Axes over which to compute the norm.
    keepdims: Whether to keep the original number of dimensions.
    eps: Epsilon value to avoid NaN gradients at zero.

  Returns:
    The safe norm of x.
  """
  sq = jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)
  return jnp.sqrt(jnp.maximum(sq, eps * eps))


def transpose(x: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
  """Transpose of a tensor on the last two dimensions."""
  return jnp.swapaxes(x, -1, -2)


################################################################################
# MARK: Base Class
################################################################################


class Manifold(Protocol):
  """Protocol for a Riemannian manifold.

  This class implements the Riemannian geometry of a manifold. In what follows,
  we denote by x and y points on the manifold, and by v a tangent vector at x.

  Methods:
    exp: Exponential map.
    log: Logarithm map.
    dist: Riemannian distance.
    project: Project vector v to tangent space at x.
    random_uniform: Sample from uniform distribution on the manifold.
    velocity: Velocity of the geodesic between x and y at time t.

  The exponential map pushes a point x along a tangent vector v.
  It is necessary in our Riemannian integration scheme.

  The logarithm map is the inverse of the exponential map and is used to compute
  the velocity of the geodesic between x and y at time t.

  The distance function is used to compute the distance between x and y on the
  manifold.

  The projection operator is used to project ambient vectors onto the tangent
  space at a point x on the manifold.

  It is necessary in most Riemannian architectures to ensure that
  the output of neural network layers are tangent vectors.

  The random uniform distribution is used to sample from the base distribution
  (uniform) on the manifold.

  The velocity function is used to compute the velocity of the geodesic between
  x and y at time t. It is the target of the regression problem in Riemannian
  Flow Matching.
  """

  def exp(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Exponential map."""
    ...

  def log(self, x: DataArray, y: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Logarithm map."""
    ...

  def dist(self, x: DataArray, y: DataArray) -> LossOutput:  # pyrefly: ignore[not-a-type]
    """Riemannian distance."""
    ...

  def project(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Project vector v to tangent space at x."""
    ...

  def random_uniform(self, key: PRNGKey, shape: tuple[int, ...]) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Sample from uniform distribution on the manifold."""
    ...

  def velocity(
      self,
      x: DataArray,  # pyrefly: ignore[not-a-type]
      y: DataArray,  # pyrefly: ignore[not-a-type]
      t: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Velocity of the geodesic between x and y at time t."""
    ...


################################################################################
# MARK: Common Manifold Methods
################################################################################


def dist_sq(manifold: Manifold, x: DataArray, y: DataArray) -> LossOutput:  # pyrefly: ignore[not-a-type]
  """Squared Riemannian distance."""
  return jnp.square(manifold.dist(x, y))


def geodesic(
    manifold: Manifold,
    x: DataArray,  # pyrefly: ignore[not-a-type]
    y: DataArray,  # pyrefly: ignore[not-a-type]
    t: TimeArray,  # pyrefly: ignore[not-a-type]
) -> DataArray:  # pyrefly: ignore[not-a-type]
  """Geodesic between x and y at time t in [0, 1].

  A geodesic is the generalization of a straight line to a curved manifold.
  In our case, where a logarithm map is available, a geodesic is given by:

    geodesic(x, y, t) = exp(x, t * log(x, y))

  Indeed, the derivative of this expression with respect to t is:

    d/dt geodesic(x, y, t) = log(x, y)

  which does not depend on t. See Definition 10.16 in
  https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf for
  more details.

  Args:
    manifold: The manifold to compute the geodesic on.
    x: Points on the manifold.
    y: Points on the manifold.
    t: Time/integration steps. At t=0, geodesic = x. At t=1, geodesic = y.

  Returns:
    Geodesic between x and y at time t.
  """
  return manifold.exp(x, t * manifold.log(x, y))


################################################################################
# MARK: Sphere
################################################################################


class Sphere(Manifold):
  """Unit hypersphere manifold S^d embedded in ambient Euclidean space.

  This class implements the Riemannian geometry of the unit hypersphere. Points
  x reside on the surface of the sphere (i.e., have a Euclidean norm of 1).
  Tangent vectors v at a point x are orthogonal to x in the ambient space
  (i.e., the dot product <x, v> = 0).

  In all manifold operations, inputs are assumed to be batched. Operations are
  vectorized over the first dimension.

  Shape Conventions:
      x, y (points): Shape `(bsz, *ambient_shape)`
      v (tangent vectors): Shape `(bsz, *ambient_shape)`
      t (time/integration steps): Shape `(bsz, 1)` or broadcastable.

  Where `bsz` is the batch size and `*ambient_shape` represents the dimensions
  of the ambient Euclidean space (which flattens to d+1 dimensions).

  We refer to
  https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere/ for
  more details.
  """

  def exp(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Exponential map on S^d.

    Compute the exponential map on the sphere according to the formula:

      exp(x, v) = cos(|v|) * x + sin(|v|) * v / |v|

    Args:
      x: Points on the sphere.
      v: Tangent vectors at x.

    Returns:
      Points on the sphere obtained by moving from x in the direction of v.
    """
    non_batch_axes = tuple(range(1, x.ndim))
    v_norm = safe_norm(v, axis=non_batch_axes, keepdims=True)
    # recall that unnormalized_sinc(x) = sin(x) / x
    return jnp.cos(v_norm) * x + unnormalized_sinc(v_norm) * v

  def log(self, x: DataArray, y: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Logarithm map on S^d.

    Compute the logarithm map on the sphere according to the formula:

      log(x, y) = (theta / sin(theta)) * (y - cos(theta) x) ,

    where cos(theta) = <x, y>.

    Note: The log map is ill-defined when x and y are nearly antipodal
    (theta -> pi), as the geodesic direction becomes ambiguous. The
    epsilon-clipping of cos_theta prevents theta from reaching exactly pi,
    but results near the antipodal point may be inaccurate.

    Args:
      x: Points on the sphere.
      y: Points on the sphere.

    Returns:
      Tangent vectors at x pointing towards y.
    """
    non_batch_axes = tuple(range(1, x.ndim))
    cos_theta = jnp.sum(x * y, axis=non_batch_axes, keepdims=True)
    cos_theta = jnp.clip(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)
    # theta away from pi, since in those cases the log map is ill-defined.
    theta = jnp.arccos(cos_theta)
    return (y - cos_theta * x) / unnormalized_sinc(theta)

  def dist(self, x: DataArray, y: DataArray) -> LossOutput:  # pyrefly: ignore[not-a-type]
    """Distance on S^d.

    Compute the distance on the sphere according to the formula:

      dist(x, y) = arccos(<x, y>)

    Args:
      x: Points on the sphere.
      y: Points on the sphere.

    Returns:
      Distance between x and y.
    """
    non_batch_axes = tuple(range(1, x.ndim))
    cos_theta = jnp.sum(x * y, axis=non_batch_axes)
    cos_theta = jnp.clip(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)
    return jnp.arccos(cos_theta)

  def project(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Project vector v to tangent space at x.

    Compute the projection of v onto the tangent space at x according to the
    formula:

      project(x, v) = v - <x, v> x

    Args:
      x: Points on the sphere.
      v: Tangent vectors at x.

    Returns:
      The projection of v onto the tangent space at x.
    """
    non_batch_axes = tuple(range(1, x.ndim))
    return v - jnp.sum(x * v, axis=non_batch_axes, keepdims=True) * x

  def random_uniform(self, key: PRNGKey, shape: tuple[int, ...]) -> DataArray:  # pyrefly: ignore[not-a-type]
    # Samples from N(0, I) and normalizes on the sphere.
    non_batch_axes = tuple(range(1, len(shape)))
    z = jax.random.normal(key, shape)
    z_norm = safe_norm(z, axis=non_batch_axes, keepdims=True)
    return z / z_norm

  def velocity(
      self,
      x: DataArray,  # pyrefly: ignore[not-a-type]
      y: DataArray,  # pyrefly: ignore[not-a-type]
      t: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Velocity of the geodesic between x and y at time t.

    Compute the velocity of the geodesic between x and y at time t according to
    the formula:

      v(t) = -|v|sin(t|v|) x + cos(t|v|) v

    Args:
      x: Points on the sphere.
      y: Points on the sphere.
      t: Time/integration steps.

    Returns:
      Velocity of the geodesic between x and y at time t.
    """
    non_batch_axes = tuple(range(1, x.ndim))

    # Calculate theta safely (this replaces recalculating the norm of v)
    cos_theta = jnp.sum(x * y, axis=non_batch_axes, keepdims=True)
    cos_theta = jnp.clip(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)
    # theta away from pi, since in those cases the log map is ill-defined.
    theta = jnp.arccos(cos_theta)

    # Calculate v using the sinc trick
    v = (y - cos_theta * x) / unnormalized_sinc(theta)

    # theta is mathematically equivalent to |v|
    return -theta * jnp.sin(t * theta) * x + jnp.cos(t * theta) * v


################################################################################
# MARK: Rotation matrices SO(3)
################################################################################


def _hat(v: Array['*batch 3']) -> Array['*batch 3 3']:  # pyrefly: ignore[not-a-type]
  """Hat map: R^3 -> so(3). Maps a 3D vector to a skew-symmetric matrix.

  Note that the operation is vectorized over the first dimension.
  More precisely, given v = (x, y, z), returns:
    [[0, -z, y],
      [z, 0, -x],
      [-y, x, 0]]

  Note that this is the inverse of the vee map.
  We refer to
  https://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf
  for more details. In particular, Equation (2.4) in Chapter 2.1.

  Args:
    v: Tangent vectors in R^3.

  Returns:
    Skew-symmetric matrix in so(3).
  """
  x, y, z = v[..., 0], v[..., 1], v[..., 2]
  zeros = jnp.zeros_like(x)
  return jnp.stack(
      [
          jnp.stack([zeros, -z, y], axis=-1),
          jnp.stack([z, zeros, -x], axis=-1),
          jnp.stack([-y, x, zeros], axis=-1),
      ],
      axis=-2,
  )


def _vee(omega: Array['*batch 3 3']) -> Array['*batch 3']:  # pyrefly: ignore[not-a-type]
  """Vee map: so(3) -> R^3. Maps a skew-symmetric matrix back to a 3D vector.

  Note that the operation is vectorized over the first dimension.
  More precisely, given:
    [[0, -z, y],
      [z, 0, -x],
      [-y, x, 0]]
  returns (x, y, z).

  Note that this is the inverse of the hat map.
  We refer to
  https://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf
  for more details. In particular, Equation (2.4) in Chapter 2.1.

  Args:
    omega: Skew-symmetric matrix in so(3).

  Returns:
    Tangent vectors in R^3.
  """
  return jnp.stack(
      [omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]], axis=-1
  )


class SO3(Manifold):
  """Special Orthogonal Group SO(3) of 3x3 rotation matrices.

  This class implements the Riemannian geometry of the SO(3) manifold. Points
  R reside on the manifold (i.e., they are 3x3 orthogonal matrices with a
  determinant of +1). Tangent vectors V at a point R reside in the tangent
  space T_R SO(3), which is represented by R * Omega, where Omega is a 3x3
  skew-symmetric matrix in the Lie algebra so(3).
  """

  def exp(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Exponential map on SO(3).

    Computes the exact exponential map using Rodrigues' rotation formula.

    First, we compute the skew matrix Omega from the tangent vector v and the
    current rotation matrix x by

      Omega = x^T @ v

    We then extract the angle theta from Omega, as well as the axis k.
    The angle theta is the norm of the matrix Omega, decomposed in the Lie group
    basis (see _vee above). The axis k is the matrix Omega normalized by the
    angle.

    Then, we use Rodrigues' rotation formula to compute the exponential

      exp(Omega) = I + sinc(theta) * Omega + cosc(theta) * Omega @ Omega

    where cosc(theta) = (1 - cos(theta)) / theta^2.
    Finally, the exponential map is given by

      exp(x, v) = x @ exp(Omega) = x @ exp(x^T @ v)

    Args:
      x: Points on the manifold.
      v: Tangent vectors at x.

    Returns:
      Points on the manifold obtained by moving from x in the direction of v.
    """
    omega_mat = jnp.matmul(transpose(x), v)
    omega = _vee(omega_mat)

    # Safe norm to protect the backward pass from NaNs at zero
    angle = safe_norm(omega, axis=(-1,), keepdims=True)

    i_mat = jnp.eye(3)

    exp_mat = (
        i_mat
        + unnormalized_sinc(angle)[..., None] * omega_mat
        + unnormalized_cosc(angle)[..., None] * jnp.matmul(omega_mat, omega_mat)
    )
    return jnp.matmul(x, exp_mat)

  def log(self, x: DataArray, y: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Logarithm map on SO(3).

    This is the inverse of the exponential map.
    To compute the tangent vector v, so that exp(x, v) = y, we recall that we
    have:

      y = x @ exp(Omega) = x @ exp(x^T @ v)

    So in order to find v, we first compute Omega by:

      Omega = log(x^T @ y)

    To compute the logarithm map, we invert Rodrigues' rotation formula.
    We recall Rodrigues' rotation formula to compute the exponential

      exp(Omega) = I + sinc(theta) * Omega + cosc(theta) * Omega @ Omega

    We recall that Omega = theta k, where theta is the angle of the rotation,
    and k is a unit skew-symmetric matrix, representing the axis of rotation.
    Note here that k has eigenvalues (0, i, -i) and so k @ k has trace -2.
    Hence, we can compute theta by:

      cos(theta) = trace(exp(Omega)) - 1.0) / 2

    From this, we can compute theta and then Omega by:

      Omega = (exp(Omega) - exp(-Omega)^T) / (2 * sinc(theta))

    So, with this we have Omega = x^T @ v. Finally, we compute v by:

      v = x @ Omega

    Args:
      x: Points on the manifold.
      y: Points on the manifold.

    Returns:
      Tangent vectors at x pointing towards y.
    """
    exp_omega_mat = jnp.matmul(transpose(x), y)
    cos_theta = (jnp.trace(exp_omega_mat, axis1=-1, axis2=-2) - 1.0) / 2.0
    cos_theta = jnp.clip(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)
    theta = jnp.arccos(cos_theta)[..., None, None]

    # Prevent pi singularity division by zero while preserving gradients
    sinc_theta = unnormalized_sinc(theta)

    omega_mat = (1.0 / (2.0 * sinc_theta)) * (
        exp_omega_mat - transpose(exp_omega_mat)
    )
    return jnp.matmul(x, omega_mat)

  def dist(self, x: DataArray, y: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Computes the shortest geodesic distance between rotations x and y.

    Let y = exp(x, v) = x @ exp(Omega). Then the distance is given by theta,
    where Omega = theta k, and k is a unit skew-symmetric matrix. We can compute
    theta by:

      cos(theta) = trace(x^T @ y) - 1.0) / 2.0
      theta = arccos(cos(theta))

    Args:
      x: Points on the manifold.
      y: Points on the manifold.

    Returns:
      Distance between x and y.
    """
    exp_omega_mat = jnp.matmul(transpose(x), y)
    cos_theta = (jnp.trace(exp_omega_mat, axis1=-1, axis2=-2) - 1.0) / 2.0
    cos_theta = jnp.clip(cos_theta, -1.0 + EPSILON, 1.0 - EPSILON)
    # theta away from pi, since in those cases the distance is ill-defined.
    return jnp.arccos(cos_theta)

  def project(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Project ambient matrix v to tangent space at x.

    Project the ambient matrix v to the tangent space at x by first shifting v
    to the Lie algebra so(3) by:

      Omega = x^T @ v

    and then computing the skew matrix skew_Omega by:

      skew_Omega = 0.5 * (Omega - Omega^T)

    Finally, we project v to the tangent space at x by:

      project(x, v) = x @ skew_Omega

    Args:
      x: Points on the manifold.
      v: Ambient matrix.

    Returns:
      Tangent vectors at x from projecting v onto the tangent space at x.
    """
    omega_mat = jnp.matmul(transpose(x), v)
    skew_omega_mat = 0.5 * (omega_mat - jnp.swapaxes(omega_mat, -1, -2))
    return jnp.matmul(x, skew_omega_mat)

  def random_uniform(self, key: PRNGKey, shape: tuple[int, ...]) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Haar measure on SO(3).

    Samples rotation matrices uniformly via quaternions.
    To do this, we first sample a unit quaternion (w, x, y, z) from the uniform
    distribution on the 4D unit sphere (S^3). Then, we convert the quaternion
    to a rotation matrix. The obtained rotation matrix is guaranteed to be
    uniformly distributed on SO(3), since the mapping from S^3 to SO(3) given
    by quaternion conjugation is a group isomorphism (double-cover of SO(3) by
    S^3)

    Args:
      key: PRNG key.
      shape: Shape of the desired output.

    Returns:
      Rotation matrices on SO(3).
    """
    quat_shape = shape[:-2] + (4,)

    q = jax.random.normal(key, quat_shape)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    row0 = jnp.stack(
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)], axis=-1
    )
    row1 = jnp.stack(
        [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)], axis=-1
    )
    row2 = jnp.stack(
        [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)], axis=-1
    )

    return jnp.stack([row0, row1, row2], axis=-2)

  def velocity(
      self,
      x: DataArray,  # pyrefly: ignore[not-a-type]
      y: DataArray,  # pyrefly: ignore[not-a-type]
      t: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Velocity of the geodesic between x and y at time t.

    This computes the time derivative (tangent vector) along the shortest path
    (geodesic) connecting rotation matrix x to rotation matrix y on the SO(3)
    manifold.

    The geodesic curve from x to y is parameterized by time t (where t=0 is x
    and t=1 is y) and defined by the Lie group exponential map:

        x(t) = x @ exp(t * Omega)

    where Omega is the skew-symmetric matrix in the Lie algebra so(3)
    representing the relative rotation axis and angle from x to y.

    The initial tangent vector at x pointing towards y is found via the
    logarithm map:

        v = log_x(y) = x @ Omega

    From this, we can extract the local Lie algebra element Omega by:

        Omega = x^T @ v

    The point on the geodesic at time t is found via the exponential map by
    scaling the tangent vector by t:

        x(t) = exp_x(t * v) = x @ exp(t * Omega)

    The velocity at time t is the time derivative of the geodesic curve x(t):

        dx(t)/dt = d/dt [x @ exp(t * Omega)]
                  = x @ exp(t * Omega) @ Omega
                  = x(t) @ Omega

    Args:
      x: Starting points on the manifold.
      y: Target points on the manifold.
      t: Time array for interpolation.

    Returns:
      Velocity (tangent vectors) at the points x(t) along the geodesics.
      Shape `(bsz, 3, 3)`.
    """
    # Compute the tangent vector at x pointing to y: v = x @ Omega
    v = self.log(x, y)

    # Find the point on the geodesic at time t: x(t) = x @ exp(t * Omega)
    xt = self.exp(x, t * v)

    # Extract the Lie algebra skew-symmetric matrix: Omega = x^T @ v
    omega = jnp.matmul(transpose(x), v)

    # Compute the velocity vector at x(t): dx(t)/dt = x(t) @ Omega
    return jnp.matmul(xt, omega)


################################################################################
# MARK: Torus
################################################################################


class Torus(Manifold):
  """T-dimensional Torus [0, 1]^d with periodic boundary conditions."""

  def exp(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    return (x + v) % 1.0

  def log(self, x: DataArray, y: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Shortest displacement on the torus."""
    return (y - x + 0.5) % 1.0 - 0.5

  def dist(self, x: DataArray, y: DataArray) -> LossOutput:  # pyrefly: ignore[not-a-type]
    return jnp.linalg.norm(self.log(x, y), axis=-1)

  def project(self, x: DataArray, v: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    return v  # Tangent space is R^d

  def random_uniform(self, key: DataArray, shape: tuple[int, ...]) -> DataArray:  # pyrefly: ignore[not-a-type]
    return jax.random.uniform(key, shape)

  def velocity(
      self,
      x: DataArray,  # pyrefly: ignore[not-a-type]
      y: DataArray,  # pyrefly: ignore[not-a-type]
      t: DataArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    # Geodesics on the flat torus are straight lines (with periodic wrapping),
    # so the velocity is constant and independent of time t.
    del t  # Unused.
    return self.log(x, y)
