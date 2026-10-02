# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complex Schur decomposition (and sqrtm, which uses it) on MPS.

schur had no MPS lowering. Complex input now lowers to Hessenberg reduction
plus Wilkinson-shifted QR sweeps on float32 (re, im) parts. Schur vectors are
only unique up to phases and eigenvalue order, so the decomposition is checked
through triangularity, unitarity, reconstruction and the eigenvalue set;
sqrtm (unique principal root) is compared with CPU directly.

Real input lowers to the real Schur form (Francis double-shift QR, then
LAPACK dlanv2 standardisation of the 2x2 blocks): T is quasi-upper-triangular,
each nonzero subdiagonal belongs to a 2x2 block with equal diagonal entries and
b * c < 0 (a complex-conjugate pair), and Z is orthogonal. jax.scipy.linalg.funm
depends on this form (schur(a) + rsf2csf), so it is compared with CPU.
"""

import jax
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from jax import numpy as jnp
from jax._src.lax import linalg as lax_linalg  # schur primitive wrapper

_RNG = np.random.default_rng(61)


def _c(shape, scale=1.0):
    return ((_RNG.standard_normal(shape) + 1j * _RNG.standard_normal(shape)) * scale).astype(np.complex64)


def _on(platform, fn, *xs, jit=True):
    f = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, f(*[jnp.asarray(x) for x in xs]))


def _same_eigenvalues(a, b, tol):
    d = np.abs(a[:, None] - b[None, :])
    assert d.min(axis=1).max() <= tol and d.min(axis=0).max() <= tol


_MATS = {
    "1x1": lambda: _c((1, 1)),
    "2x2": lambda: _c((2, 2)),
    "5x5": lambda: _c((5, 5)),
    "8x8": lambda: _c((8, 8)),
    "16x16": lambda: _c((16, 16)),
    "batched_3x6x6": lambda: _c((3, 6, 6)),
    "triangular_5x5": lambda: np.triu(_c((5, 5))).astype(np.complex64),
    "jordan_2x2": lambda: np.array([[1.0, 1.0], [0.0, 1.0]], np.complex64),
    "zero_4x4": lambda: np.zeros((4, 4), np.complex64),
}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_complex_schur_decomposition(mat, jit):
    x = _MATS[mat]()
    t, z = _on("mps", lax_linalg.schur, x, jit=jit)
    assert t.dtype == np.complex64 and z.dtype == np.complex64 and t.shape == x.shape == z.shape
    scale = max(1.0, float(np.abs(x).max()) * x.shape[-1])
    assert np.abs(np.tril(t, -1)).max(initial=0.0) <= 1e-4 * scale
    eye = np.broadcast_to(np.eye(x.shape[-1]), x.shape)
    zh = np.conj(np.swapaxes(z, -1, -2))
    np.testing.assert_allclose(zh @ z, eye, atol=3e-4)
    np.testing.assert_allclose(z @ t @ zh, x, atol=5e-4 * scale)
    for xi, ti in zip(x.reshape(-1, *x.shape[-2:]), t.reshape(-1, *t.shape[-2:]), strict=True):
        _same_eigenvalues(np.linalg.eigvals(xi.astype(np.complex128)), np.diagonal(ti), 2e-3 * scale)


def test_complex_schur_without_vectors():
    x = _c((6, 6))
    (t,) = _on("mps", lambda a: lax_linalg.schur(a, compute_schur_vectors=False), x)
    assert np.abs(np.tril(t, -1)).max() <= 1e-3
    _same_eigenvalues(np.linalg.eigvals(x.astype(np.complex128)), np.diagonal(t), 1e-2)


def test_scipy_schur_complex_output_of_real_input():
    x = _RNG.standard_normal((5, 5)).astype(np.float32)
    t, z = _on("mps", lambda a: jsl.schur(a, output="complex"), x)
    zh = np.conj(z.T)
    np.testing.assert_allclose(z @ t @ zh, x, atol=3e-3)
    assert np.abs(np.tril(t, -1)).max() <= 1e-3


@pytest.mark.parametrize("case", ["spd_f32", "shifted_c64"])
def test_sqrtm_matches_cpu(case):
    if case == "spd_f32":
        m = _RNG.standard_normal((5, 5)).astype(np.float32)
        x = (m @ m.T + 5.0 * np.eye(5)).astype(np.float32)
    else:
        x = (_c((5, 5), 0.3) + 4.0 * np.eye(5)).astype(np.complex64)
    expected = _on("cpu", jsl.sqrtm, x)
    actual = _on("mps", jsl.sqrtm, x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def _r(shape, scale=1.0):
    return (_RNG.standard_normal(shape) * scale).astype(np.float32)


def _cyclic(n):
    return np.roll(np.eye(n, dtype=np.float32), 1, axis=0)


def _symmetric(n):
    m = _r((n, n))
    return ((m + m.T) / 2).astype(np.float32)


_REAL_MATS = {
    "1x1": lambda: _r((1, 1)),
    "2x2_real_pair": lambda: np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
    "2x2_complex_pair": lambda: np.array([[1.0, -2.0], [3.0, 0.5]], np.float32),
    "2x2_rotation": lambda: np.array([[0.0, -1.0], [1.0, 0.0]], np.float32),
    "3x3_doc_example": lambda: np.array([[0.0, 3.0, 3.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], np.float32),
    "5x5": lambda: _r((5, 5)),
    "8x8": lambda: _r((8, 8)),
    "16x16": lambda: _r((16, 16)),
    "batched_3x6x6": lambda: _r((3, 6, 6)),
    "symmetric_6x6": lambda: _symmetric(6),
    "triangular_5x5": lambda: np.triu(_r((5, 5))).astype(np.float32),
    "cyclic_5x5": lambda: _cyclic(5),
    "cyclic_6x6": lambda: _cyclic(6),
    "zero_4x4": lambda: np.zeros((4, 4), np.float32),
}


def _check_real_schur(x, t, z):
    n = x.shape[-1]
    assert t.dtype == np.float32 and z.dtype == np.float32 and t.shape == x.shape == z.shape
    scale = max(1.0, float(np.abs(x).max()) * n)
    eye = np.broadcast_to(np.eye(n), x.shape)
    zt = np.swapaxes(z, -1, -2)
    np.testing.assert_allclose(zt @ z, eye, atol=3e-4)
    np.testing.assert_allclose(z @ t @ zt, x, atol=5e-4 * scale)
    for xi, ti in zip(x.reshape(-1, n, n), t.reshape(-1, n, n), strict=True):
        assert np.all(np.tril(ti, -2) == 0), "entries below the first subdiagonal must be exactly zero"
        sub = np.diagonal(ti, offset=-1)
        for j in np.nonzero(sub)[0]:  # block rows/cols (j, j + 1)
            assert j == 0 or sub[j - 1] == 0, f"consecutive nonzero subdiagonals at {j}"
            np.testing.assert_allclose(ti[j, j], ti[j + 1, j + 1], atol=1e-5 * scale)
            assert ti[j, j + 1] * ti[j + 1, j] < 0, "a 2x2 block must hold a complex-conjugate pair"
        _same_eigenvalues(
            np.linalg.eigvals(xi.astype(np.float64)), np.linalg.eigvals(ti.astype(np.float64)), 2e-3 * scale
        )


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_REAL_MATS))
def test_real_schur_decomposition(mat, jit):
    x = _REAL_MATS[mat]()
    t, z = _on("mps", lax_linalg.schur, x, jit=jit)
    _check_real_schur(x, t, z)


def test_real_schur_without_vectors():
    x = _r((6, 6))
    (t,) = _on("mps", lambda a: lax_linalg.schur(a, compute_schur_vectors=False), x)
    assert np.all(np.tril(t, -2) == 0)
    _same_eigenvalues(np.linalg.eigvals(x.astype(np.float64)), np.linalg.eigvals(t.astype(np.float64)), 1e-2)


def test_scipy_schur_default_real_output():
    x = _r((7, 7))
    t, z = _on("mps", jsl.schur, x)
    _check_real_schur(x, t, z)


def _spd_scaled():
    m = _r((5, 5))
    return ((m @ m.T + 5.0 * np.eye(5)) / 10.0).astype(np.float32)


_FUNM_CASES = {
    "spd_exp": (_spd_scaled, jnp.exp),
    "general_sin_cos": (lambda: _r((5, 5), 0.5), lambda v: jnp.sin(v) + 2 * jnp.cos(v)),
    "doc_example_exp": (lambda: np.array([[1.0, 2.0], [3.0, 4.0]], np.float32) / 4, jnp.exp),
}


@pytest.mark.parametrize("case", list(_FUNM_CASES))
def test_funm_real_matches_cpu(case):
    make, func = _FUNM_CASES[case]
    x = make()
    expected = _on("cpu", lambda a: jsl.funm(a, func), x)
    actual = _on("mps", lambda a: jsl.funm(a, func), x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
