# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strided / general reduce_window must run on MPS and match XLA CPU.

Moving sums and products with a stride > 1 (any rank, any axis) and rank-1
strided max/min were rejected with "reduce_window: unsupported pattern". They
now use an exact pad-with-init, strided-slice-and-fold lowering.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_RNG = np.random.default_rng(0)
_X1 = _RNG.standard_normal((17,)).astype(np.float32)
_X2 = _RNG.standard_normal((4, 17)).astype(np.float32)
_X3 = _RNG.standard_normal((2, 4, 17)).astype(np.float32)
_NAN = _X2.copy()
_NAN[1, 5] = np.nan


def _rw(init, op, win, strides, pad="VALID", dil=None):
    return lambda a: lax.reduce_window(a, init, op, win, strides, pad, window_dilation=dil)


_CASES = {
    "r1_sum_w3_s2": (_rw(0.0, lax.add, (3,), (2,)), _X1),
    "r1_max_w3_s2": (_rw(-jnp.inf, lax.max, (3,), (2,)), _X1),
    "r1_min_w2_s2": (_rw(jnp.inf, lax.min, (2,), (2,)), _X1),
    "r1_sum_w4_s3_same": (_rw(0.0, lax.add, (4,), (3,), "SAME"), _X1),
    "r2_sum_w3_s2": (_rw(0.0, lax.add, (1, 3), (1, 2)), _X2),
    "r2_sum_axis0_w2_s2": (_rw(0.0, lax.add, (2, 1), (2, 1)), _X2),
    "r2_sum_w2x3_s2x2": (_rw(0.0, lax.add, (2, 3), (2, 2)), _X2),
    "r2_mul_w3_s2": (_rw(1.0, lax.mul, (1, 3), (1, 2)), _X2),
    "r2_sum_dilated": (_rw(0.0, lax.add, (1, 3), (1, 2), "VALID", (1, 2)), _X2),
    "r2_sum_nonzero_init_padded": (_rw(0.5, lax.add, (1, 3), (1, 2), ((0, 0), (1, 2))), _X2),
    "r2_max_nan": (_rw(-jnp.inf, lax.max, (1, 3), (1, 2)), _NAN),
    "r3_sum_w3_s2_last": (_rw(0.0, lax.add, (1, 1, 3), (1, 1, 2)), _X3),
    "i32_sum_w3_s2": (_rw(0, lax.add, (1, 3), (1, 2)), (_X2 * 1000).astype(np.int32)),
    "i8_sum_wraps": (_rw(np.int8(0), lax.add, (3,), (2,)), np.full((9,), 100, np.int8)),
    "f16_sum_w3_s2": (_rw(np.float16(0), lax.add, (1, 3), (1, 2)), _X2.astype(np.float16)),
}


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return np.asarray(func(jnp.asarray(x)))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("case", list(_CASES))
def test_strided_reduce_window_matches_cpu(case, jit):
    fn, x = _CASES[case]
    expected = _run(fn, x, "cpu", jit)
    actual = _run(fn, x, "mps", jit)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        tol = 2e-3 if expected.dtype == np.float16 else 1e-5
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_strided_moving_sum_grad_matches_cpu(jit):
    fn = jax.grad(lambda a: (lax.reduce_window(a, 0.0, lax.add, (1, 3), (1, 2), "VALID") ** 2).sum())
    np.testing.assert_allclose(_run(fn, _X2, "mps", jit), _run(fn, _X2, "cpu", jit), rtol=1e-5, atol=1e-5)
