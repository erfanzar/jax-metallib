# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integer dot/matmul must match CPU instead of aborting the process.

MPSGraph matmul rejects integer operands with a fatal assertion (exit 134),
so each case runs in a child process.
"""

import os
import subprocess
import sys

import pytest

SCRIPT = """
import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

dtype, case, mode = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.default_rng(0)
big = dtype in ("int32", "uint32")
lo, hi = (-(1 << 20), 1 << 20) if big else (-300, 300)


def arr(shape):
    return rng.integers(lo, hi, size=shape, dtype=np.int64).astype(dtype)


if case == "mm":
    args = (arr((5, 7)), arr((7, 3)))
    fn = lambda a, b: a @ b
elif case == "vec":
    args = (arr((7,)), arr((7, 3)))
    fn = lambda a, b: a @ b
elif case == "batched":
    args = (arr((2, 5, 7)), arr((2, 7, 3)))
    fn = lambda a, b: jnp.einsum("bij,bjk->bik", a, b)
elif case == "multi_batch":
    args = (arr((2, 3, 4, 5)), arr((2, 3, 5, 6)))
    fn = lambda a, b: jnp.einsum("abij,abjk->abik", a, b)
elif case == "self":
    args = (arr((5, 7)),)
    fn = lambda a: a @ a.T
else:
    args = (arr((5, 7)), arr((7, 3)))
    fn = lambda a, b: lax.dot(a, b, preferred_element_type=jnp.int32)

if mode == "jit":
    fn = jax.jit(fn)
with jax.default_device(jax.devices("cpu")[0]):
    expected = np.asarray(fn(*map(jnp.asarray, args)))
with jax.default_device(jax.devices("mps")[0]):
    actual = np.asarray(fn(*map(jnp.asarray, args)))
assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
np.testing.assert_array_equal(actual, expected)
"""


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize("case", ["mm", "vec", "batched", "multi_batch", "self", "preferred_i32"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "int32", "uint32"])
def test_integer_dot_matches_cpu(dtype, case, mode):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu,mps"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, dtype, case, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] or result.stdout
