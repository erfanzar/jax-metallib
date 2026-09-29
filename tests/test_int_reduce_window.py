# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integer reduce_window max/min must match CPU exactly.

Integer extrema windows previously raised "reduce_window: unsupported
pattern" (the pooling and sliding paths were float-only). Cases run in child
processes in case an MPSGraph lowering aborts.
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

dtype, reducer, case, mode = sys.argv[1:5]
info = np.iinfo(dtype)
rng = np.random.default_rng(0)
shape = (3, 5, 7) if case == "rank3" else (6, 9)
x = rng.integers(int(info.min), int(info.max), size=shape, endpoint=True, dtype=np.int64)
x.flat[::5] = info.max
x.flat[1::7] = info.min
if info.bits == 32:
    x.flat[2::11] = int(info.max) - 1
x = x.astype(dtype)
op = lax.max if reducer == "max" else lax.min
init = info.min if reducer == "max" else info.max

CASES = {
    "same_1x3": ((1, 3), (1, 1), "SAME", None),
    "valid_2x2": ((2, 2), (1, 1), "VALID", None),
    "strided_2x3": ((2, 3), (2, 2), ((1, 0), (0, 2)), None),
    "dilated": ((2, 2), (1, 1), ((1, 1), (2, 2)), (2, 3)),
    "rank3": ((1, 2, 3), (1, 1, 2), "SAME", None),
    "nonidentity_init": ((1, 3), (1, 1), "SAME", None),
}
window, strides, padding, wdil = CASES[case]
if case == "nonidentity_init":
    init = int(info.max) // 2 if reducer == "max" else int(info.min) // 2 + 1


def fn(a):
    return lax.reduce_window(a, np.asarray(init, dtype), op, window, strides, padding, window_dilation=wdil)


if mode == "jit":
    fn = jax.jit(fn)
with jax.default_device(jax.devices("cpu")[0]):
    expected = np.asarray(fn(jnp.asarray(x)))
with jax.default_device(jax.devices("mps")[0]):
    actual = np.asarray(fn(jnp.asarray(x)))
assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
np.testing.assert_array_equal(actual, expected)
"""


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize("case", ["same_1x3", "valid_2x2", "strided_2x3", "dilated", "rank3", "nonidentity_init"])
@pytest.mark.parametrize("reducer", ["max", "min"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "int32", "uint32"])
def test_integer_reduce_window_extrema_match_cpu(dtype, reducer, case, mode):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu,mps"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, dtype, reducer, case, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] or result.stdout
