# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Float argmax/argmin must match XLA:CPU exactly, including NaN semantics.

XLA returns the index of the first NaN when a slice contains NaN, otherwise the
first index of the extremum (-0 == +0, so ties go to the lower index). The MPS
lowering moves a non-last reduction axis to the end. For float rows that were
moved, or that are <= 2048 long, it arg-reduces an int32 order key built from
the bit pattern (NaN -> INT32_MAX/MIN, -0 and +0 -> 0) and gathers the value.
Longer last-axis float rows use the uint8 NaN-mask arg-reduction plus a
one-element gather per slice. JAX_METALLIB_DISABLE_ARGREDUCE_KEY=1 restores the
direct lowering for every slice. These cases pin down both routes.
"""

import os
import subprocess
import sys

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_SPECIAL = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 3.5], np.float32)
_DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": jnp.bfloat16}
_FNS = {"argmax": jnp.argmax, "argmin": jnp.argmin}


def _edge(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    special = rng.random(shape) < 0.7
    x[special] = rng.choice(_SPECIAL, size=int(special.sum()))
    return x


def _run(fn, x):
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("op", list(_FNS))
@pytest.mark.parametrize("shape,axis", [((512, 37), -1), ((512, 37), 0), ((4, 37, 9), 1), ((300, 1), -1)])
def test_argreduce_special_values(dtype, op, shape, axis):
    x = _edge(shape).astype(_DTYPES[dtype])
    _run(lambda a: _FNS[op](a, axis=axis), x)


@pytest.mark.parametrize("op", list(_FNS))
def test_argreduce_all_nan_and_no_nan_rows(op):
    x = np.random.default_rng(1).standard_normal((6, 16)).astype(np.float32)
    x[0] = np.nan
    x[1, 5] = np.nan
    x[1, 9] = np.nan
    x[2] = 2.0
    x[3, :4] = [np.inf, np.nan, -np.inf, np.nan]
    x[4, -1] = np.nan
    _run(lambda a: _FNS[op](a, axis=-1), x)


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("op", list(_FNS))
@pytest.mark.parametrize(
    "shape,axis",
    [((8, 2048), -1), ((8, 2049), -1), ((2049, 8), 0), ((5, 33, 17), 0), ((5, 33, 17), 1), ((3, 600, 300), 1)],
)
def test_argreduce_key_gate_and_moved_axes(dtype, op, shape, axis):
    x = _edge(shape, seed=3).astype(_DTYPES[dtype])
    _run(lambda a: _FNS[op](a, axis=axis), x)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int8, np.int16])
@pytest.mark.parametrize("op", list(_FNS))
@pytest.mark.parametrize("shape,axis", [((300, 40), 0), ((5, 33, 17), 1), ((7, 9, 11), 0)])
def test_argreduce_int_moved_axis(dtype, op, shape, axis):
    info = np.iinfo(dtype)
    x = np.random.default_rng(4).integers(info.min, info.max, size=shape, endpoint=True, dtype=np.int64)
    x[..., :2] = info.max if op == "argmax" else info.min  # ties go to the lower index
    _run(lambda a: _FNS[op](a, axis=axis), x.astype(dtype))


def _value_index_reduce(x, axis, is_max):
    cmp = lax.gt if is_max else lax.lt

    def body(a, b):
        (av, ai), (bv, bi) = a, b
        pick = jnp.logical_or(cmp(av, bv), av != av)
        pick_i = jnp.logical_or(pick, jnp.logical_and(av == bv, ai < bi))
        return lax.select(pick, av, bv), lax.select(pick_i, ai, bi)

    axis = axis % x.ndim  # broadcasted_iota and lax.reduce need a non-negative axis
    init = jnp.array(-np.inf if is_max else np.inf, x.dtype)
    idx = lax.broadcasted_iota(jnp.int32, x.shape, axis)
    return lax.reduce((x, idx), (init, jnp.int32(0)), body, (axis,))


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("is_max", [True, False])
@pytest.mark.parametrize("shape,axis", [((64, 300), -1), ((64, 3000), -1), ((300, 40), 0), ((5, 33, 17), 1)])
def test_variadic_value_index_reduce(dtype, is_max, shape, axis):
    x = _edge(shape, seed=5).astype(_DTYPES[dtype])
    fn = lambda a: _value_index_reduce(a, axis, is_max)  # noqa: E731
    ev, ei = (np.asarray(t) for t in jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        av, ai = (np.asarray(t) for t in jax.jit(fn)(jnp.asarray(x)))
    np.testing.assert_array_equal(ai, ei)
    assert av.dtype == ev.dtype
    np.testing.assert_array_equal(av.astype(np.float64), ev.astype(np.float64))  # NaN == NaN here


_OPT_OUT_SCRIPT = r"""
import jax, numpy as np
from jax import numpy as jnp
rng = np.random.default_rng(6)
x = rng.standard_normal((300, 40)).astype(np.float32)
x[rng.random(x.shape) < 0.05] = np.nan
x[3] = -0.0
for fn in (jnp.argmax, jnp.argmin):
    for ax in (0, 1):
        want = np.asarray(jax.jit(lambda a: fn(a, axis=ax), backend="cpu")(x))
        with jax.default_device(jax.devices("mps")[0]):
            got = np.asarray(jax.jit(lambda a: fn(a, axis=ax))(jnp.asarray(x)))
        assert np.array_equal(got, want), (fn.__name__, ax)
print("OK")
"""


def test_argreduce_key_opt_out_matches_cpu():
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_DISABLE_ARGREDUCE_KEY="1")
    proc = subprocess.run([sys.executable, "-c", _OPT_OUT_SCRIPT], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]


@pytest.mark.parametrize("op", list(_FNS))
def test_argreduce_wide_rows_single_nan(op):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((256, 4096)).astype(np.float32)
    rows = rng.choice(256, size=64, replace=False)
    x[rows, rng.integers(0, 4096, size=64)] = np.nan
    _run(lambda a: _FNS[op](a, axis=-1), x)
