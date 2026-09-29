# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integer argmax/argmin must match CPU instead of aborting the process.

MPSNDArrayReduction asserted on uint32 arg-reductions ("For a UInt32 source,
destination must also be UInt32"), so each case runs in a child process.
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

dtype, op, mode = sys.argv[1], sys.argv[2], sys.argv[3]
info = np.iinfo(dtype)
x = np.random.default_rng(0).integers(int(info.min), int(info.max), size=(4, 32), endpoint=True, dtype=np.int64)
x[:, 3] = info.max
x[:, 7] = info.max  # ties must resolve to the lowest index
x[:, 11] = info.min
x[:, 13] = info.min
if info.bits == 32:
    x[:, 5] = x[:, 3] - 1  # near-tie in the lowest bit
x = x.astype(dtype)

fns = {
    "argmax": lambda a: jnp.argmax(a, axis=-1),
    "argmin": lambda a: jnp.argmin(a, axis=0),
    "max_and_arg": lambda a: (jnp.max(a, axis=-1), jnp.argmax(a, axis=-1)),
}
fn = fns[op]
if mode == "jit":
    fn = jax.jit(fn)
with jax.default_device(jax.devices("cpu")[0]):
    expected = jax.tree.map(np.asarray, fn(jnp.asarray(x)))
with jax.default_device(jax.devices("mps")[0]):
    actual = jax.tree.map(np.asarray, fn(jnp.asarray(x)))
jax.tree.map(np.testing.assert_array_equal, actual, expected)
"""


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize("op", ["argmax", "argmin", "max_and_arg"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32"])
def test_integer_arg_reduce_matches_cpu(dtype, op, mode):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu,mps"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, dtype, op, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] or result.stdout
