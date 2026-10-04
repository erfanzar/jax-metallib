# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""8/16-bit dynamic_slice inside loop bodies must match XLA:CPU without widening.

HandleDynamicSlice widens f16/int8/uint8/int16/uint16 operands to 32 bits to
avoid an eager-only segfault. Inside a scan that cast covered the whole operand
every iteration (GPT-2 medium f16 over stacked layer weights: 9.8 ms vs 3.0 ms),
so loop bodies now slice the narrow operand directly. These cases cover stacked
reads, clamped (out-of-range) starts and integer dtypes; the eager path keeps
the widening and is covered by tests/test_small_dtype_slice_reverse.py.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_DTYPES = [np.float16, np.int8, np.uint8, np.int16, np.uint16]


def _data(dtype, shape):
    rng = np.random.default_rng(3)
    if np.issubdtype(dtype, np.floating):
        x = rng.standard_normal(shape).astype(dtype)
        x.reshape(-1)[::17] = np.nan
        x.reshape(-1)[5::19] = -0.0
        return x
    info = np.iinfo(dtype)
    return rng.integers(info.min, int(info.max) + 1, size=shape).astype(dtype)


def _stacked_read(w):
    # w[i] of a stacked operand, like per-layer weights in a scan.
    def body(c, i):
        return c, w[i]

    return lax.scan(body, jnp.int32(0), jnp.arange(w.shape[0]))[1]


def _clamped_rows(w):
    # Starts run past both ends of axis 0 and must clamp like XLA.
    def body(c, i):
        return c, lax.dynamic_slice(w, (i * 5 - 7, 1), (3, w.shape[1] - 2))

    return lax.scan(body, jnp.int32(0), jnp.arange(8))[1]


def _fori_sum(w):
    def body(i, acc):
        return acc + lax.dynamic_slice(w, (i, 0), (1, w.shape[1])).astype(jnp.float32).sum()

    return lax.fori_loop(0, w.shape[0], body, jnp.float32(0))


def _run(fn, x):
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    return actual, expected


def _bits_equal(actual, expected):
    view = {1: np.uint8, 2: np.uint16, 4: np.uint32}[actual.dtype.itemsize]
    np.testing.assert_array_equal(actual.view(view), expected.view(view))


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
def test_stacked_read_in_scan(dtype):
    _bits_equal(*_run(_stacked_read, _data(dtype, (6, 64, 48))))


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
def test_clamped_starts_in_scan(dtype):
    _bits_equal(*_run(_clamped_rows, _data(dtype, (12, 40))))


def test_f16_row_sum_in_fori_loop():
    x = np.random.default_rng(4).standard_normal((24, 512)).astype(np.float16)
    actual, expected = _run(_fori_sum, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
