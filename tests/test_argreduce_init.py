# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Variadic value/index reduce must include both init operands.

Use finite exact-valued inputs so a dropped init is not confused with floating
reduction order. Each case runs in a child process to isolate MPSGraph caches.
Both returned tensors are compared bitwise with XLA:CPU.
"""

import os
import subprocess
import sys

import pytest

_SCRIPT = r"""
import sys

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

name, dtype_name, kind, axis, mode, traced = sys.argv[1:]
axis = int(axis)
traced = traced == "traced"
dtype = getattr(jnp, dtype_name)
shape = (4, 4096) if mode == "wide" else (6, 9)
is_max = kind == "max"
rng = np.random.default_rng(72)
x = rng.integers(1, 17, size=shape).astype(dtype)
index_init = np.int32(3)
if mode == "neutral":
    index_init = np.int32(0)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        value_init = info.min if is_max else info.max
    else:
        value_init = -np.inf if is_max else np.inf
elif mode == "partial":
    value_init = 8
    selector = [slice(None)] * x.ndim
    selector[axis] = 0
    x[tuple(selector)] = 20 if is_max else 0
    selector[axis] = slice(None)
    selector[1 - axis] = 0
    x[tuple(selector)] = 4 if is_max else 12
elif mode == "nan":
    value_init = np.nan
elif mode == "tie":
    value_init = 8
    x[:] = 8
    index_init = np.int32(-2)
elif mode == "zero":
    value_init = -0.0
    x[:] = 0.0
    index_init = np.int32(-2)
else:
    value_init = 100 if is_max else 0
value_init = np.asarray(value_init, dtype)


def body(a, b):
    av, ai = a
    bv, bi = b
    better = lax.gt(av, bv) if is_max else lax.lt(av, bv)
    pick_value = lax.bitwise_or(better, lax.ne(av, av))
    pick_index = lax.bitwise_or(pick_value, lax.bitwise_and(lax.eq(av, bv), lax.lt(ai, bi)))
    return lax.select(pick_value, av, bv), lax.select(pick_index, ai, bi)


def reduce_with_init(a, v, i):
    idx = lax.broadcasted_iota(jnp.int32, a.shape, axis)
    return lax.reduce((a, idx), (v, i), body, (axis,))


if traced:
    fn = reduce_with_init
else:
    def fn(a, v, i):
        return reduce_with_init(a, jnp.asarray(value_init), jnp.asarray(index_init))


results = {}
for platform in ("cpu", "mps"):
    device = jax.devices(platform)[0]
    args = [jax.device_put(t, device) for t in (x, value_init, index_init)]
    results[platform] = tuple(np.asarray(t) for t in jax.jit(fn)(*args))
for label, want, got in zip(("value", "index"), results["cpu"], results["mps"], strict=True):
    assert got.shape == want.shape and got.dtype == want.dtype, (name, label, got.shape, want.shape)
    bits = np.dtype(f"uint{want.dtype.itemsize * 8}")
    np.testing.assert_array_equal(got.view(bits), want.view(bits), err_msg=f"{name}: {label}")
print("OK", name)
"""

_CASES = [
    (f"{kind}_{dtype}_{axis}_{traced}", dtype, kind, axis, "dominant", traced)
    for dtype in ("float32", "float16", "bfloat16", "int32", "uint32")
    for kind in ("max", "min")
    for axis, traced in ((1, "constant"), (0, "traced"))
]
_CASES += [
    (f"{kind}_{mode}_{traced}", "float32", kind, 1, mode, traced)
    for kind in ("max", "min")
    for mode in ("partial", "nan", "tie", "zero", "wide", "neutral")
    for traced in ("constant", "traced")
]


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_variadic_reduce_init_matches_cpu(case):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, *(str(arg) for arg in case)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout[-1000:] + proc.stderr[-3000:]
