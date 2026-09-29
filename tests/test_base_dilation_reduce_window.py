# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""reduce_window with base_dilation > 1 must run on MPS and match XLA CPU.

It was rejected with "reduce_window: base_dilations != 1 is unsupported" before
any lowering ran. XLA:CPU is the reference for how dilation holes interact with a
nonzero init value.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_RNG = np.random.default_rng(1)
_X1 = _RNG.standard_normal((9,)).astype(np.float32)
_X2 = _RNG.standard_normal((3, 7)).astype(np.float32)


def _rw(init, op, win, strides, pad="VALID", base=None, dil=None):
    return lambda a: lax.reduce_window(a, init, op, win, strides, pad, base_dilation=base, window_dilation=dil)


_CASES = {
    "r1_sum_b2": (_rw(0.0, lax.add, (3,), (1,), base=(2,)), _X1),
    "r1_sum_b3_s2": (_rw(0.0, lax.add, (4,), (2,), base=(3,)), _X1),
    "r1_max_b2": (_rw(-jnp.inf, lax.max, (3,), (1,), base=(2,)), _X1),
    "r1_min_b2_padded": (_rw(jnp.inf, lax.min, (2,), (1,), ((1, 1),), base=(2,)), _X1),
    "r1_sum_nonzero_init_b2": (_rw(0.5, lax.add, (3,), (1,), base=(2,)), _X1),
    "r1_max_nonzero_init_b2": (_rw(0.25, lax.max, (3,), (1,), base=(2,)), _X1),
    "r2_sum_b1x2": (_rw(0.0, lax.add, (1, 3), (1, 1), base=(1, 2)), _X2),
    "r2_sum_b2x2_w2x2": (_rw(0.0, lax.add, (2, 2), (1, 1), base=(2, 2)), _X2),
    "r2_max_b2x1": (_rw(-jnp.inf, lax.max, (2, 1), (1, 1), base=(2, 1)), _X2),
    "r2_mul_b1x2": (_rw(1.0, lax.mul, (1, 3), (1, 1), base=(1, 2)), _X2),
    "r2_sum_b2_dilated_window": (_rw(0.0, lax.add, (1, 2), (1, 1), base=(1, 2), dil=(1, 2)), _X2),
    "i32_sum_b2": (_rw(0, lax.add, (3,), (1,), base=(2,)), (_X1 * 1000).astype(np.int32)),
    "i32_max_b2": (_rw(np.iinfo(np.int32).min, lax.max, (3,), (1,), base=(2,)), (_X1 * 1000).astype(np.int32)),
    "f16_sum_b2": (_rw(np.float16(0), lax.add, (3,), (1,), base=(2,)), _X1.astype(np.float16)),
}


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return np.asarray(func(jnp.asarray(x)))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("case", list(_CASES))
def test_base_dilation_reduce_window_matches_cpu(case, jit):
    fn, x = _CASES[case]
    expected = _run(fn, x, "cpu", jit)
    actual = _run(fn, x, "mps", jit)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        tol = 2e-3 if expected.dtype == np.float16 else 1e-5
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, equal_nan=True)
