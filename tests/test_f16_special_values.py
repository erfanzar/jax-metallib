# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""float16 NaN, signed-zero and domain-edge values must match XLA CPU.

With MPSGraph's default optimization level, float16 subgraphs could be placed
on the Neural Engine, where `nan != nan` was False, log(0) gave -45440,
sqrt(-x) gave 0, exp(nan) gave inf and f32->f16 casts turned NaN into inf and
-0 into +0. The plugin now compiles graphs at MPSGraphOptimizationLevel0.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_VALS = np.array([np.nan, -0.0, 0.0, -0.5, 1.0, 2.0, np.inf, -np.inf], dtype=np.float32)

_OPS = {
    "isnan_cmp": lambda a: a != a,
    "isnan": jnp.isnan,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
}


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return np.asarray(func(jnp.asarray(x)))


def _assert_same(expected, actual):
    assert actual.dtype == expected.dtype
    if expected.dtype == np.bool_:
        np.testing.assert_array_equal(actual, expected)
        return
    e, a = expected.astype(np.float64), actual.astype(np.float64)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(e))
    finite = ~np.isnan(e)
    np.testing.assert_allclose(a[finite], e[finite], rtol=2e-3)
    zero = finite & (e == 0)
    np.testing.assert_array_equal(np.signbit(a[zero]), np.signbit(e[zero]))


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_f16_special_values_match_cpu(op, jit):
    x = _VALS.astype(np.float16)
    _assert_same(_run(_OPS[op], x, "cpu", jit), _run(_OPS[op], x, "mps", jit))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_f32_to_f16_cast_keeps_nan_and_negative_zero(jit):
    fn = lambda a: a.astype(jnp.float16)  # noqa: E731
    _assert_same(_run(fn, _VALS, "cpu", jit), _run(fn, _VALS, "mps", jit))
