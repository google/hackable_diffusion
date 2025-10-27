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

"""Wrapper around jaxtyping to support short-form array annotations.

I.e. Float["b n d"] instead of Float[Array, "b n d"]

Also defines a @typechecked decorator that uses the typeguard typechecker.

You can disable type checking by setting the environment variable
`JAXTYPING_DISABLE_HD` to `1`. This is dangerous and can lead to
undefined behavior! Use at your own risk.
"""

import os
from typing import Any
from absl import logging
import jaxtyping
import jaxtyping._typeguard as typeguard_jaxtyping
import typeguard

# An easy option to disable typeching for the users.
JAXTYPING_DISABLE_HD = os.environ.get("JAXTYPING_DISABLE_HD", "0")

if JAXTYPING_DISABLE_HD == "1":
  logging.warning(
      "You are disabling type checking for HD. This is dangerous and can lead"
      " to undefined behavior! Use at your own risk."
  )

jaxtyping.config.update("jaxtyping_disable", JAXTYPING_DISABLE_HD)


class ArrayAliasMeta(type):
  """Metaclass helper to create array annotations that support short-form.

  Usage:
  ```
  Float = ArrayAliasMeta("Float", jaxtyping.Float)

  def foo(x: Float["b n d"]):  # notice the single arg to Float[]
    pass
  ```
  """

  dtype: Any

  def __new__(
      mcs,
      name: str,
      jaxtyping_type: Any,
  ):
    return super().__new__(mcs, name, (), {"dtype": jaxtyping_type})

  def __init__(cls, name: str, dtype: Any):
    del name, dtype
    super().__init__(cls)

  def __instancecheck__(cls, inst: Any):
    return isinstance(inst, cls["..."])

  def __getitem__(cls, dim_spec: str):
    return cls.dtype[jaxtyping.Array, dim_spec]


def typechecked(fn):
  """Typechecking decorator."""
  return jaxtyping.jaxtyped(typechecker=typeguard_jaxtyping.typechecked)(fn)


def check_type(value: Any, expected_type: Any) -> None:
  """Check if the value is of the expected type, else raise TypeCheckError.

  Works best in conjunction with the @typechecked decorator.
  If the check is successful, the inferred dimensions are memorized, same as
  with array annotations in the function definition with @typechecked.

  Args:
    value: The value to check.
    expected_type: The expected type of the value.

  Usage:
  ```
  @typechecked
  def foo(x: Float["b n c"]):
    v = ...
    check_type(v, Float["b n d"])
    ...
  ```
  """
  typeguard.check_type(value, expected_type)


Array = ArrayAliasMeta("Array", jaxtyping.Shaped)

Bool = ArrayAliasMeta("Bool", jaxtyping.Bool)

Int = ArrayAliasMeta("Int", jaxtyping.Integer)  # Int = UInt | SInt

SInt = ArrayAliasMeta("SInt", jaxtyping.Int)
Int8 = ArrayAliasMeta("Int8", jaxtyping.Int8)
Int16 = ArrayAliasMeta("Int16", jaxtyping.Int16)
Int32 = ArrayAliasMeta("Int32", jaxtyping.Int32)
Int64 = ArrayAliasMeta("Int64", jaxtyping.Int64)

UInt = ArrayAliasMeta("UInt", jaxtyping.UInt)
UInt8 = ArrayAliasMeta("UInt8", jaxtyping.UInt8)
UInt16 = ArrayAliasMeta("UInt8", jaxtyping.UInt16)
UInt32 = ArrayAliasMeta("UInt32", jaxtyping.UInt32)
UInt64 = ArrayAliasMeta("UInt64", jaxtyping.UInt64)
Complex = ArrayAliasMeta("Complex", jaxtyping.Complex)
Complex64 = ArrayAliasMeta("Complex64", jaxtyping.Complex64)

Float = ArrayAliasMeta("Float", jaxtyping.Float)
BFloat16 = ArrayAliasMeta("BFloat16", jaxtyping.BFloat16)
Float32 = ArrayAliasMeta("Float32", jaxtyping.Float32)
Float64 = ArrayAliasMeta("Float64", jaxtyping.Float64)

Num = ArrayAliasMeta("Num", jaxtyping.Num)

Scalar = Array[""]


# Supports both old and new-style jax PRNG keys.
# See: https://docs.jax.dev/en/latest/jep/9263-typed-keys.html
PRNGKey = UInt32["2"] | jaxtyping.Key[jaxtyping.Array, ""]
