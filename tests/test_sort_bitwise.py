# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Float sort must match XLA:CPU bit for bit, including stable order of ties.

JAX's sort comparator maps -0 to +0 and every NaN to one canonical NaN before a
stable LT compare, so -0/+0 ties and NaNs with different sign/payload keep
their input order and their exact bits. The native bitonic kernel was not
stable (row 0 of an alternating +/-0 input came back [+0,-0,NaN] vs CPU
[-0,+0,NaN]); value-only checks (nan_to_num) could not see it.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_NAN_BITS = np.array([0x7FC00001, 0xFFC00002, 0x7F800003, 0xFFC00000, 0x7FC00000], np.uint32)


def _ties(rows, cols, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    x[:, ::3] = 0.0
    x[:, 1::6] = -0.0
    nan_cols = rng.choice(cols, size=min(cols, 12), replace=False)
    x[:, nan_cols] = _NAN_BITS[np.arange(nan_cols.size) % _NAN_BITS.size].view(np.float32)
    x[:, 2] = np.inf
    x[:, 4] = -np.inf
    return x


def _alternating_zero(rows, cols):
    x = np.zeros((rows, cols), np.float32)
    x[:, ::2] = -0.0
    x[0, 5] = np.nan
    return x


_CASES = {
    "ties_512x300": lambda: _ties(512, 300, 0),
    "ties_64x1000": lambda: _ties(64, 1000, 1),
    "ties_16x4096": lambda: _ties(16, 4096, 2),
    "ties_8x8192": lambda: _ties(8, 8192, 3),
    "alt_zero_4096x256": lambda: _alternating_zero(4096, 256),
    "random_4096x256": lambda: np.random.default_rng(4).standard_normal((4096, 256)).astype(np.float32),
}


def _run(fn, x):
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    return actual, expected


def _bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    bad = np.any(actual.view(np.uint32) != expected.view(np.uint32), axis=-1)
    assert not bad.any(), f"{int(bad.sum())} rows differ bitwise, first {np.flatnonzero(bad)[:5]}"


@pytest.mark.parametrize("case", list(_CASES))
def test_row_sort_bitwise_matches_cpu(case):
    _bitwise(*_run(lambda a: lax.sort(a, dimension=-1), _CASES[case]()))


@pytest.mark.parametrize("case", ["ties_512x300", "alt_zero_4096x256"])
def test_descending_sort_bitwise_matches_cpu(case):
    _bitwise(*_run(lambda a: jnp.sort(a, axis=-1, descending=True), _CASES[case]()))


@pytest.mark.parametrize("n", [1000, 5000, 20000, 100000])
def test_1d_sort_bitwise_matches_cpu(n):
    # Covers the 1-D native / CPU vDSP sort paths.
    x = _ties(1, n, 5)[0]
    _bitwise(*_run(jnp.sort, x))
