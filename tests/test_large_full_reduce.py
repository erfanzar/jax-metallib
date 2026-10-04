# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Large full (to-scalar) reductions that stay in MPSGraph must match XLA:CPU.

TreeReduce used to build one slice + reduce per 256-element chunk and chain the
chunk results serially (~12K graph ops at 1M elements, int32 sum ~59 ms vs
~1.2 ms f32). It now reduces through a reshape tree; these cases cover lengths
that are not multiples of the window, integer wraparound, extrema and NaN.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_RNG = np.random.default_rng(11)


def _f32(n):
    return _RNG.standard_normal((n,)).astype(np.float32)


def _f32_with_nan(n, pos):
    x = _f32(n)
    x[pos] = np.nan
    return x


def _i32(n):
    return _RNG.integers(-(2**31), 2**31 - 1, size=(n,), dtype=np.int64).astype(np.int32)


_CASES = {
    "count_neg_1000003": (lambda a: jnp.sum(a < 0, dtype=jnp.int32), lambda: _f32(1_000_003)),
    "count_nonzero_65537": (lambda a: jnp.count_nonzero(a > 0.5), lambda: _f32(65_537)),
    "i32_sum_wraps_300001": (lambda a: jnp.sum(a), lambda: _i32(300_001)),
    "i32_max_1000000": (lambda a: jnp.max(a), lambda: _i32(1_000_000)),
    "i32_min_2d_700x1300": (lambda a: jnp.min(a), lambda: _i32(910_000).reshape(700, 1300)),
    "i32_prod_wraps_5000": (lambda a: jnp.prod(a | 1), lambda: _i32(5_000)),
    "bool_any_all_200000": (lambda a: jnp.stack([jnp.any(a > 4.5), jnp.all(a < 6.0)]), lambda: _f32(200_000)),
    "f32_max_nan_2049": (lambda a: jnp.max(a), lambda: _f32_with_nan(2049, 1777)),
    "f32_sum_3000": (lambda a: jnp.sum(a), lambda: _f32(3_000)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_large_full_reduce_matches_cpu(case):
    fn, make = _CASES[case]
    x = make()
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual, expected)
