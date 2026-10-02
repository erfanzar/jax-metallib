# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""General (non-symmetric) eigendecomposition on MPS.

eig had no MPS lowering. Eigenvalues now come from the complex Schur form;
eigenvectors from triangular substitution on T (right) and T^H (left), mapped
back through the Schur vectors and normalised like LAPACK geev (unit 2-norm,
largest component real). Eigenvalue order is implementation-defined, so
values are compared as sets and vectors after matching eigenvalues.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax._src.lax import linalg as lax_linalg  # eig primitive wrapper

_RNG = np.random.default_rng(71)


def _c(shape):
    return (_RNG.standard_normal(shape) + 1j * _RNG.standard_normal(shape)).astype(np.complex64)


def _sym(n):
    m = _RNG.standard_normal((n, n))
    return ((m + m.T) / 2).astype(np.float32)


_MATS = {
    "c64_5x5": lambda: _c((5, 5)),
    "c64_8x8": lambda: _c((8, 8)),
    "f32_5x5": lambda: _RNG.standard_normal((5, 5)).astype(np.float32),
    "f32_sym_6x6": lambda: _sym(6),
    "c64_batched_3x4x4": lambda: _c((3, 4, 4)),
    "c64_1x1": lambda: _c((1, 1)),
    "c64_triangular_5x5": lambda: (np.triu(_c((5, 5))) + np.diag(np.arange(5.0) * 2)).astype(np.complex64),
}


def _on(platform, fn, *xs, jit=True):
    f = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, f(*[jnp.asarray(x) for x in xs]))


def _scale(x):
    return max(1.0, float(np.abs(x).max()) * x.shape[-1])


def _flat(a):
    return a.reshape(-1, *a.shape[-2:]) if a.ndim > 2 else a[None]


def _same_set(a, b, tol):
    d = np.abs(a[:, None] - b[None, :])
    assert d.min(axis=1).max() <= tol and d.min(axis=0).max() <= tol


@pytest.mark.parametrize("mat", list(_MATS))
def test_eigvals_match_cpu(mat):
    x = _MATS[mat]()
    expected = _on("cpu", jnp.linalg.eigvals, x)
    actual = _on("mps", jnp.linalg.eigvals, x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    for a, e in zip(actual.reshape(-1, x.shape[-1]), expected.reshape(-1, x.shape[-1]), strict=True):
        _same_set(a, e, 2e-3 * _scale(x))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_eig_right_vectors(mat, jit):
    x = _MATS[mat]()
    w, v = _on("mps", jnp.linalg.eig, x, jit=jit)
    assert v.dtype == np.complex64 and v.shape == x.shape
    for xi, wi, vi in zip(_flat(x), w.reshape(-1, x.shape[-1]), _flat(v), strict=True):
        np.testing.assert_allclose(xi @ vi, vi * wi[None, :], atol=2e-3 * _scale(x))
        np.testing.assert_allclose(np.linalg.norm(vi, axis=0), 1.0, atol=1e-4)
        big = np.argmax(np.abs(vi), axis=0)
        assert np.abs(vi[big, np.arange(vi.shape[1])].imag).max() <= 1e-5


@pytest.mark.parametrize("mat", ["c64_5x5", "f32_5x5", "c64_batched_3x4x4"])
def test_eig_left_vectors(mat):
    x = _MATS[mat]()

    def fn(a):
        return lax_linalg.eig(a, compute_left_eigenvectors=True, compute_right_eigenvectors=True)

    w, vl, vr = _on("mps", fn, x)
    for xi, wi, li, ri in zip(_flat(x), w.reshape(-1, x.shape[-1]), _flat(vl), _flat(vr), strict=True):
        lh = np.conj(li.T)
        np.testing.assert_allclose(lh @ xi, wi[:, None] * lh, atol=2e-3 * _scale(x))
        np.testing.assert_allclose(xi @ ri, ri * wi[None, :], atol=2e-3 * _scale(x))
        np.testing.assert_allclose(np.linalg.norm(li, axis=0), 1.0, atol=1e-4)


@pytest.mark.parametrize("mat", ["c64_5x5", "f32_5x5"])
def test_eig_vectors_match_cpu_after_matching(mat):
    # Distinct eigenvalues + geev normalisation (unit norm, largest component
    # real) fix each vector up to a real sign: LAPACK leaves the sign of a real
    # eigenvector of a real matrix unspecified. Align by the unit phase and
    # require that phase to be real (+/-1).
    x = _MATS[mat]()
    w_c, v_c = _on("cpu", jnp.linalg.eig, x)
    w_m, v_m = _on("mps", jnp.linalg.eig, x)
    order = np.argmin(np.abs(w_m[:, None] - w_c[None, :]), axis=1)
    assert sorted(order.tolist()) == list(range(x.shape[-1]))
    np.testing.assert_allclose(w_m, w_c[order], atol=1e-3)
    v_c = v_c[:, order]
    inner = np.sum(np.conj(v_c) * v_m, axis=0)
    phase = inner / np.abs(inner)
    np.testing.assert_allclose(np.abs(phase.imag), 0.0, atol=1e-3)
    np.testing.assert_allclose(v_m, v_c * phase[None, :], atol=2e-3)


def _shifted(make):
    # Well-separated eigenvalues keep the eigenvalue/eigenvector derivatives
    # (1 / (l_i - l_j) terms) well conditioned.
    x = make()
    return (x + np.diag(np.arange(x.shape[-1], dtype=np.float32) * 1.5)).astype(x.dtype)


_GRAD_MATS = {
    "c64_4x4": lambda: _shifted(lambda: _c((4, 4))),
    "f32_4x4": lambda: _shifted(lambda: _RNG.standard_normal((4, 4)).astype(np.float32)),
}


@pytest.mark.parametrize("mat", list(_GRAD_MATS))
def test_eigvals_grad_matches_cpu(mat):
    # eig's JVP rule is JAX's; it runs on the primal computed by the MPS lowering.
    x = _GRAD_MATS[mat]()

    def loss(a):
        w = jnp.linalg.eigvals(a)
        return jnp.sum(jnp.real(w) ** 2 + 0.5 * jnp.imag(w) ** 2)

    expected = _on("cpu", jax.grad(loss), x)
    actual = _on("mps", jax.grad(loss), x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3 * _scale(x))


@pytest.mark.parametrize("mat", list(_GRAD_MATS))
def test_eigvec_grad_matches_cpu(mat):
    # |v|^4 is invariant to each eigenvector's phase, which LAPACK and the MPS
    # lowering may choose differently; the opt-in eigenvector derivative must
    # then agree.
    x = _GRAD_MATS[mat]()

    def loss(a):
        w, v = lax_linalg.eig(a, compute_left_eigenvectors=False, enable_eigvec_derivs=True)
        return jnp.sum(jnp.abs(v) ** 4) + jnp.sum(jnp.abs(w))

    expected = _on("cpu", jax.grad(loss), x)
    actual = _on("mps", jax.grad(loss), x)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


def _rotation_blocks(k):
    # Block diagonal of [[a, b], [-b, a]] conjugated by a random orthogonal matrix:
    # every eigenvalue is a complex pair a +/- i b.
    rng = np.random.default_rng(5)
    d = np.zeros((2 * k, 2 * k))
    for j in range(k):
        a, b = rng.uniform(-2, 2), rng.uniform(0.5, 2)
        d[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[a, b], [-b, a]]
    q, _ = np.linalg.qr(rng.standard_normal((2 * k, 2 * k)))
    return (q @ d @ q.T).astype(np.float32)


def _real_spectrum(n):
    # Similar to diag(1..n): every eigenvalue is real and well separated.
    rng = np.random.default_rng(6)
    s = rng.standard_normal((n, n)) + n * np.eye(n)
    return (s @ np.diag(np.arange(1.0, n + 1)) @ np.linalg.inv(s)).astype(np.float32)


_REAL_EIGVALS_MATS = {
    "f32_32x32": lambda: np.random.default_rng(32).standard_normal((32, 32)).astype(np.float32),
    "f32_64x64": lambda: np.random.default_rng(64).standard_normal((64, 64)).astype(np.float32),
    "f32_batched_3x8x8": lambda: np.random.default_rng(8).standard_normal((3, 8, 8)).astype(np.float32),
    "f32_real_spectrum_10": lambda: _real_spectrum(10),
    "f32_complex_pairs_12": lambda: _rotation_blocks(6),
    "f32_2x2_complex": lambda: np.array([[1.0, 2.0], [-3.0, 1.0]], np.float32),
}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_REAL_EIGVALS_MATS))
def test_real_eigvals_from_real_schur(mat, jit):
    # Real input, eigenvalues only: taken from the real Schur form's 1x1 / 2x2
    # blocks instead of the complex Schur form. Pairs follow dhseqr: adjacent,
    # exact conjugates, positive imaginary part first.
    x = _REAL_EIGVALS_MATS[mat]()
    w = _on("mps", jnp.linalg.eigvals, x, jit=jit)
    assert w.dtype == np.complex64 and w.shape == x.shape[:-1]
    n = x.shape[-1]
    for xi, wi in zip(_flat(x), w.reshape(-1, n), strict=True):
        ref = np.linalg.eigvals(xi.astype(np.float64))
        _same_set(wi, ref, 5e-5 * max(1.0, float(np.abs(ref).max())) * max(1.0, n / 16))
        j = 0
        while j < n:
            if wi[j].imag != 0:
                assert j + 1 < n and wi[j].imag > 0 and wi[j + 1] == np.conj(wi[j]), wi
                j += 2
            else:
                j += 1
    if mat == "f32_real_spectrum_10":
        assert np.all(w.imag == 0), w
    if mat.startswith("f32_complex_pairs") or mat == "f32_2x2_complex":
        assert np.all(w.imag != 0), w


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_REAL_EIGVALS_MATS))
def test_real_eig_vectors_from_real_schur(mat, jit):
    # Real input with eigenvectors: complex Schur form built from the real Schur
    # form (rsf2csf) instead of the complex QR iteration. Checked like geev:
    # A V = V diag(w), unit 2-norm columns, largest component real.
    x = _REAL_EIGVALS_MATS[mat]()
    w, v = _on("mps", jnp.linalg.eig, x, jit=jit)
    n = x.shape[-1]
    assert w.dtype == np.complex64 and w.shape == x.shape[:-1]
    assert v.dtype == np.complex64 and v.shape == x.shape
    for xi, wi, vi in zip(_flat(x), w.reshape(-1, n), _flat(v), strict=True):
        x64, v64 = xi.astype(np.float64), vi.astype(np.complex128)
        resid = np.abs(x64 @ v64 - v64 * wi[None, :]).max()
        assert resid <= 1e-5 * _scale(xi), resid
        np.testing.assert_allclose(np.linalg.norm(v64, axis=0), 1.0, atol=1e-5)
        big = vi[np.argmax(np.abs(vi), axis=0), np.arange(n)]
        assert np.abs(big.imag).max() <= 1e-6 * np.abs(big).max(), big
        ref = np.linalg.eigvals(x64)
        _same_set(wi, ref, 5e-5 * max(1.0, float(np.abs(ref).max())) * max(1.0, n / 16))
