# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""stablehlo.gather must move 32-bit integers exactly.

MPSGraph's gather lost precision for int32/uint32 values above 2**24 (off by
1-2), both inside and outside loop bodies, while float32 gathers were exact.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _values(dtype):
    base = np.random.default_rng(2).integers(-1000, 1000, size=(64, 16), dtype=np.int64) + (1 << 25)
    if dtype == np.uint32:
        base = base + (1 << 31)
    return base.astype(dtype)


def _take_rows(x, order):
    return jnp.take(x, order, axis=0)


def _take_scalar(x, order):
    return jnp.take(x, order[5], axis=0)


def _take_cols(x, order):
    return jnp.take(x, order % 16, axis=1)


def _take_in_scan(x, order):
    def step(carry, t):
        return carry, jnp.take(x, order[t], axis=0)

    return lax.scan(step, jnp.int32(0), jnp.arange(x.shape[0], dtype=jnp.int32))[1]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32], ids=["int32", "uint32"])
@pytest.mark.parametrize("fn", [_take_rows, _take_scalar, _take_cols, _take_in_scan], ids=lambda f: f.__name__)
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_gather_32bit_integers_exact(dtype, fn, jit):
    values = _values(dtype)
    order = np.random.default_rng(3).permutation(64).astype(np.int32)
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(jnp.asarray(values), jnp.asarray(order)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(jnp.asarray(values), jnp.asarray(order)))
    np.testing.assert_array_equal(actual, expected)
