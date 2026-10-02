# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native-only linalg ops inside control-flow regions must run on MPS.

jax.scipy.linalg.expm solves its Pade system inside a runtime stablehlo.case,
so the native-only triangular_solve landed in a graph region and failed with
"Native op encountered during graph building". triangular_solve now has a
graph fallback used only inside such regions; top-level solves stay native.
"""

import itertools

import jax
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_RNG = np.random.default_rng(17)


def _mat(n, scale, batch=()):
    return (_RNG.standard_normal((*batch, n, n)) * scale).astype(np.float32)


def _run(fn, *xs, rtol=1e-4, atol=1e-4):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*xs))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(*[jnp.asarray(x) for x in xs]))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


_EXPM = {
    "small_norm_4": lambda: _mat(4, 0.05),
    "mid_norm_6": lambda: _mat(6, 0.5),
    "large_norm_5": lambda: _mat(5, 2.0),
    "batched_3x4": lambda: _mat(4, 0.7, (3,)),
}


@pytest.mark.parametrize("case", list(_EXPM))
def test_expm_matches_cpu(case):
    _run(jsl.expm, _EXPM[case](), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "left_side,lower,transpose_a,unit_diagonal",
    list(itertools.product([True, False], repeat=4)),
    ids=lambda v: str(int(v)),
)
def test_triangular_solve_in_cond(left_side, lower, transpose_a, unit_diagonal):
    # Both triangles hold data: only the requested one may be read.
    n = 5
    a = (_mat(n, 0.3) + 4.0 * np.eye(n)).astype(np.float32)
    b = _RNG.standard_normal((n, 3) if left_side else (3, n)).astype(np.float32)

    def fn(a, b, p):
        def solve():
            return lax.linalg.triangular_solve(
                a, b, left_side=left_side, lower=lower, transpose_a=transpose_a, unit_diagonal=unit_diagonal
            )

        return lax.cond(p > 0, solve, lambda: b)

    _run(fn, a, b, np.float32(1.0))


def test_batched_triangular_solve_in_cond():
    a = (_mat(6, 0.3, (2,)) + 4.0 * np.eye(6)).astype(np.float32)
    b = _RNG.standard_normal((2, 6, 4)).astype(np.float32)

    def fn(a, b, p):
        return lax.cond(p > 0, lambda: lax.linalg.triangular_solve(a, b, left_side=True, lower=True), lambda: b)

    _run(fn, a, b, np.float32(1.0))


def _spd(n, batch=()):
    m = _mat(n, 1.0, batch)
    return (m @ np.swapaxes(m, -1, -2) + n * np.eye(n)).astype(np.float32)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("batch", [(), (3,)], ids=["single", "batched"])
def test_cholesky_in_cond(lower, batch):
    # Garbage in the unused triangle must not leak into the factor.
    a = _spd(6, batch)
    rows, cols = np.indices((6, 6))
    unused = rows < cols if lower else rows > cols
    a = np.where(unused, 99.0, a).astype(np.float32)

    def fn(a, p):
        def chol():
            return (
                lax.linalg.cholesky(a, symmetrize_input=False)
                if lower
                else jnp.swapaxes(lax.linalg.cholesky(jnp.swapaxes(a, -1, -2), symmetrize_input=False), -1, -2)
            )

        return lax.cond(p > 0, chol, lambda: a)

    _run(fn, a, np.float32(1.0))


def test_cho_solve_in_fori_loop():
    a = _spd(5)
    b = _RNG.standard_normal((5, 2)).astype(np.float32)

    def fn(a, b):
        return lax.fori_loop(0, 2, lambda i, x: jsl.cho_solve(jsl.cho_factor(a + i), x), b)

    _run(fn, a, b)


@pytest.mark.parametrize("shape", [(4, 4), (6, 4)], ids=["square", "tall"])
def test_polar_matches_cpu(shape):
    x = _RNG.standard_normal(shape).astype(np.float32)
    _run(lambda a: jsl.polar(a)[0], x, rtol=1e-3, atol=1e-3)


def test_triangular_solve_in_fori_loop():
    a = (np.tril(_mat(5, 0.3)) + 3.0 * np.eye(5)).astype(np.float32)
    b = _RNG.standard_normal((5, 2)).astype(np.float32)

    def fn(a, b):
        return lax.fori_loop(0, 3, lambda i, x: lax.linalg.triangular_solve(a, x, left_side=True, lower=True), b)

    _run(fn, a, b)
