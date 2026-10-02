# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complex triangular_solve / cholesky and everything built on them must run on MPS.

The native triangular_solve and cholesky handlers are float-only and returned
nil for complex64, so complex solve, inv, cholesky, cho_solve and expm failed.
Complex inputs now lower to a substitution / column Cholesky on float32
(real, imag) parts; real inputs keep the native path.
"""

import itertools

import jax
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_RNG = np.random.default_rng(29)


def _c(shape, scale=1.0):
    return ((_RNG.standard_normal(shape) + 1j * _RNG.standard_normal(shape)) * scale).astype(np.complex64)


def _well(n, batch=()):
    return (_c((*batch, n, n), 0.3) + 3.0 * np.eye(n)).astype(np.complex64)


def _hpd(n, batch=()):
    m = _c((*batch, n, n))
    return (m @ np.conj(np.swapaxes(m, -1, -2)) + n * np.eye(n)).astype(np.complex64)


def _run(fn, *xs, jit=True, tol=1e-4):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*xs))
    f = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(f(*[jnp.asarray(x) for x in xs]))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "left_side,lower,transpose_a,conjugate_a,unit_diagonal",
    list(itertools.product([True, False], repeat=5)),
    ids=lambda v: str(int(v)),
)
def test_complex_triangular_solve_variants(left_side, lower, transpose_a, conjugate_a, unit_diagonal):
    # Both triangles hold data: only the requested one may be read.
    a = _well(5)
    b = _c((5, 3) if left_side else (3, 5))

    def fn(a, b):
        return lax.linalg.triangular_solve(
            a,
            b,
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
            conjugate_a=conjugate_a,
            unit_diagonal=unit_diagonal,
        )

    _run(fn, a, b)


def test_complex_batched_triangular_solve():
    _run(lambda a, b: lax.linalg.triangular_solve(a, b, left_side=True, lower=True), _well(6, (2,)), _c((2, 6, 4)))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("batch", [(), (3,)], ids=["single", "batched"])
def test_complex_solve_and_inv(batch, jit):
    a = _well(6, batch)
    _run(jnp.linalg.solve, a, _c((*batch, 6, 2)), jit=jit, tol=1e-3)
    _run(jnp.linalg.inv, a, jit=jit, tol=1e-3)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("batch", [(), (2,)], ids=["single", "batched"])
def test_complex_cholesky(batch, jit):
    _run(jnp.linalg.cholesky, _hpd(5, batch), jit=jit, tol=1e-3)


def test_complex_cho_solve():
    _run(lambda a, b: jsl.cho_solve(jsl.cho_factor(a), b), _hpd(5), _c((5, 2)), tol=1e-3)


@pytest.mark.parametrize("scale", [0.05, 0.5, 2.0], ids=["small", "mid", "large"])
def test_complex_expm(scale):
    _run(jsl.expm, _c((4, 4), scale), tol=2e-3)


def test_real_triangular_solve_unchanged():
    a = (np.tril(_RNG.standard_normal((6, 6))) + 3.0 * np.eye(6)).astype(np.float32)
    b = _RNG.standard_normal((6, 3)).astype(np.float32)
    _run(lambda a, b: lax.linalg.triangular_solve(a, b, left_side=True, lower=True), a, b)
