# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""8/16-bit dynamic_slice and reverse must match CPU instead of crashing.

On MPS, eager dynamic_slice with runtime start indices (which is how eager
`a[0]`, `lax.slice` and `squeeze` lower) segfaulted the process for
float16/int8/uint8 operands; 1-D int16 reverse under JIT returned wrong values;
and float16 `jnp.sort(descending=True)` (ascending sort + reverse) came back
unreversed under JIT. Each case runs in a child process.
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

dtype, case, mode = sys.argv[1:4]
x = (np.random.default_rng(3).standard_normal((4, 32)) * 50).astype(np.float32).astype(dtype)
CASES = {
    "getitem0": lambda a: a[0],
    "static_slice": lambda a: lax.slice(a, (1, 2), (3, 20)),
    "dyn_slice": lambda a: lax.dynamic_slice(a, (1, 3), (2, 8)),
    "squeeze": lambda a: lax.squeeze(a[:1], (0,)),
    "flip_1d": lambda a: jnp.flip(a[0]),
    "rev_2d": lambda a: lax.rev(a, (1,)),
    "sort_desc": lambda a: jnp.sort(a, axis=-1, descending=True),
}
fn = CASES[case]
if mode == "jit":
    fn = jax.jit(fn)
res = {}
for p in ("cpu", "mps"):
    with jax.default_device(jax.devices(p)[0]):
        res[p] = np.asarray(fn(jax.device_put(jnp.asarray(x), jax.devices(p)[0])))
assert res["cpu"].dtype == res["mps"].dtype, (res["cpu"].dtype, res["mps"].dtype)
np.testing.assert_array_equal(res["mps"].astype(np.float64), res["cpu"].astype(np.float64))
"""


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize("case", ["getitem0", "static_slice", "dyn_slice", "squeeze", "flip_1d", "rev_2d", "sort_desc"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int8", "uint8", "int16", "uint16"])
def test_small_dtype_slice_and_reverse_match_cpu(dtype, case, mode):
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
