# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sort and scatter must move 32-bit integers exactly.

MPSGraph gather (used to apply sort permutations and to reorder scatter
updates) lost int32/uint32 precision above 2**24.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _big(dtype, shape, seed):
    base = np.random.default_rng(seed).integers(-1000, 1000, size=shape, dtype=np.int64) + (1 << 25)
    if dtype == np.uint32:
        base = base + (1 << 31)
    return base.astype(dtype)


def _sort(x, keys):
    return jnp.sort(x, axis=-1)


def _argsort_take(x, keys):
    return jnp.take_along_axis(x, jnp.argsort(keys, axis=-1), axis=-1)


def _sort_key_val(x, keys):
    return lax.sort_key_val(keys, x, dimension=-1)[1]


def _scatter_add(x, keys):
    idx = jnp.argsort(keys[0]) % 8
    return jnp.zeros((8,), x.dtype).at[idx].add(x[0])


def _scatter_set(x, keys):
    idx = jnp.argsort(keys[0])
    return jnp.zeros((x.shape[1],), x.dtype).at[idx].set(x[0])


def _scatter_min(x, keys):
    idx = jnp.argsort(keys[0]) % 8
    return jnp.full((8,), jnp.iinfo(x.dtype).max, x.dtype).at[idx].min(x[0])


def _scatter_max(x, keys):
    idx = jnp.argsort(keys[0]) % 8
    return jnp.full((8,), jnp.iinfo(x.dtype).min, x.dtype).at[idx].max(x[0])


def _top_k(x, keys):
    # Near-ties differ only in low bits; ties must resolve lowest index first.
    x = x.at[:, ::3].set(x[:, :1] + x.dtype.type(1))
    values, indices = lax.top_k(x, 5)
    return values, indices


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_sort_mixed_sign_int32_exact(descending, jit):
    # sortWithTensor returned inexact values for large negative int32 even when
    # ascending; the value sort now uses argSort plus an exact gather.
    values = _big(np.int32, (4, 32), 6)
    values[:, ::2] = -values[:, ::2]

    def fn(x):
        return jnp.sort(x, axis=-1, descending=descending), lax.top_k(x, 5)[0]

    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("cpu")[0]):
        expected = jax.tree.map(np.asarray, func(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = jax.tree.map(np.asarray, func(jnp.asarray(values)))
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32], ids=["int32", "uint32"])
@pytest.mark.parametrize(
    "fn",
    [_sort, _argsort_take, _sort_key_val, _scatter_add, _scatter_set, _scatter_min, _scatter_max, _top_k],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_sort_scatter_32bit_integers_exact(dtype, fn, jit):
    values = _big(dtype, (4, 32), 4)
    keys = np.random.default_rng(5).standard_normal((4, 32)).astype(np.float32)
    if fn is _scatter_add:
        # Keep sums inside 32 bits: 4 addends per bucket.
        values = (values // 8).astype(dtype)
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("cpu")[0]):
        expected = jax.tree.map(np.asarray, func(jnp.asarray(values), jnp.asarray(keys)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = jax.tree.map(np.asarray, func(jnp.asarray(values), jnp.asarray(keys)))
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
