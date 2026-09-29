# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scalar-update scatter (``x.at[i, j].set(v)``) on rank >= 2 operands.

These lowered to a stablehlo.scatter with ``updates_shape=[]`` and failed with
"scatter: scalar scatter is not supported yet" for every dtype.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_CASES = {
    "set_2d": lambda a: a.at[1, 2].set(a.dtype.type(5)),
    "add_2d": lambda a: a.at[1, 2].add(a.dtype.type(5)),
    "mul_2d": lambda a: a.at[3, 7].multiply(a.dtype.type(3)),
    "max_2d": lambda a: a.at[1, 2].max(a.dtype.type(100)),
    "min_2d": lambda a: a.at[0, 3].min(a.dtype.type(2)),
    "set_3d": lambda a: a.reshape(2, 2, 8).at[1, 0, 3].set(a.dtype.type(5)),
    "set_neg_idx": lambda a: a.at[-1, -1].set(a.dtype.type(5)),
    "set_oob_dropped": lambda a: a.at[9, 2].set(a.dtype.type(5), mode="drop"),
    "chained": lambda a: a.at[0, 1].set(a.dtype.type(7)).at[2, 3].add(a.dtype.type(1)),
}


def _in_scan(a):
    def step(buf, t):
        return buf.at[t, t].set(a.dtype.type(9)), None

    return lax.scan(step, a, jnp.arange(4, dtype=jnp.int32))[0]


_CASES["in_scan_diag"] = _in_scan


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32", "int8", "uint32"])
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_scalar_scatter_matches_cpu(case, dtype, jit):
    x = np.arange(32).reshape(4, 8).astype(np.int64)
    if dtype == "int8":
        x = x + 100  # exercise 8-bit wraparound on add/multiply
    x = jnp.asarray(x).astype(dtype)
    fn = jax.jit(_CASES[case]) if jit else _CASES[case]
    res = {}
    for p in ("cpu", "mps"):
        d = jax.devices(p)[0]
        with jax.default_device(d):
            res[p] = np.asarray(fn(jax.device_put(x, d)))
    assert res["mps"].dtype == res["cpu"].dtype
    np.testing.assert_array_equal(res["mps"].astype(np.float64), res["cpu"].astype(np.float64))
