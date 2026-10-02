# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complex Hermitian eigh and complex singular values on MPS.

The Jacobi lowerings used to reject complex input outright. eigh now runs a
Hermitian Jacobi on (real, imag) float32 parts; singular values come from the
real embedding [[B, -C], [C, B]]. Eigenvalues and singular values are compared
with CPU; eigenvectors are only defined up to a phase (and a unitary rotation in
degenerate eigenspaces), so they are checked through orthonormality and
reconstruction. Complex SVD with U/V uses a complex one-sided Jacobi and fills
zero-singular-value columns from the orthogonal complement.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp


def _cplx(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _herm(n, batch=(), seed=1):
    m = _cplx((*batch, n, n), seed)
    return ((m + np.conj(np.swapaxes(m, -1, -2))) / 2).astype(np.complex64)


def _degenerate():
    q, _ = np.linalg.qr(_cplx((4, 4), 7))
    return (q @ np.diag([2.0, 2.0, -1.0, 5.0]) @ np.conj(q.T)).astype(np.complex64)


_MATS = {
    "n2": _herm(2),
    "n5": _herm(5),
    "n8": _herm(8),
    "n17": _herm(17),
    "batched_3x6": _herm(6, (3,)),
    "degenerate": _degenerate(),
}


def _on(platform, fn, *args, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, func(*map(jnp.asarray, args)))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_complex_eigvalsh_matches_cpu(mat, jit):
    x = _MATS[mat]
    expected = _on("cpu", jnp.linalg.eigvalsh, x, jit=jit)
    actual = _on("mps", jnp.linalg.eigvalsh, x, jit=jit)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_complex_eigh_vectors_reconstruct(mat, jit):
    x = _MATS[mat]
    w, v = _on("mps", jnp.linalg.eigh, x, jit=jit)
    assert v.dtype == np.complex64 and w.dtype == np.float32
    vh = np.conj(np.swapaxes(v, -1, -2))
    eye = np.broadcast_to(np.eye(x.shape[-1], dtype=np.complex64), x.shape)
    np.testing.assert_allclose(vh @ v, eye, atol=2e-4)
    np.testing.assert_allclose((v * w[..., None, :]) @ vh, x, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_complex_eigh_reads_requested_triangle(lower):
    # Non-Hermitian input with a complex diagonal: only the requested triangle
    # is used and the imaginary part of the diagonal is ignored (as in LAPACK).
    # Eigenvectors carry an arbitrary per-column phase, so compare eigenvalues
    # with CPU and reconstruct the Hermitian matrix the triangle defines.
    x = _cplx((5, 5), 4)
    rows, cols = np.indices((5, 5))
    tri = rows >= cols if lower else rows <= cols
    h = np.where(tri, x, np.conj(x.T)).astype(np.complex64)
    h[np.diag_indices(5)] = h.diagonal().real

    def fn(a):
        return jax.lax.linalg.eigh(a, lower=lower, symmetrize_input=False)

    _, w_cpu = _on("cpu", fn, x, jit=True)
    v, w = _on("mps", fn, x, jit=True)
    np.testing.assert_allclose(w, w_cpu, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose((v * w[None, :]) @ np.conj(v.T), h, rtol=2e-4, atol=2e-4)


_SVD_SHAPES = {"5x5": (5, 5), "7x4": (7, 4), "4x7": (4, 7), "batched_3x6x5": (3, 6, 5)}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("shape", list(_SVD_SHAPES))
def test_complex_svdvals_match_cpu(shape, jit):
    x = _cplx(_SVD_SHAPES[shape], 11)

    def fn(a):
        return jnp.linalg.svd(a, compute_uv=False)

    expected = _on("cpu", fn, x, jit=jit)
    actual = _on("mps", fn, x, jit=jit)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_complex_matrix_norm_and_rank_match_cpu():
    x = _cplx((6, 5), 12)
    low_rank = (_cplx((5, 2), 13) @ _cplx((2, 6), 14)).astype(np.complex64)

    for fn, arg in ((lambda a: jnp.linalg.norm(a, 2), x), (jnp.linalg.matrix_rank, low_rank)):
        expected = _on("cpu", fn, arg, jit=True)
        actual = _on("mps", fn, arg, jit=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _low_rank(m, n, r, seed):
    return (_cplx((m, r), seed) @ _cplx((r, n), seed + 1)).astype(np.complex64)


_UV_MATS = {
    "5x5": lambda: _cplx((5, 5), 21),
    "7x4": lambda: _cplx((7, 4), 22),
    "4x7": lambda: _cplx((4, 7), 23),
    "batched_3x6x5": lambda: _cplx((3, 6, 5), 24),
    "rank2_6x5": lambda: _low_rank(6, 5, 2, 25),
    "rank2_4x7": lambda: _low_rank(4, 7, 2, 27),
    "zero_3x4": lambda: np.zeros((3, 4), np.complex64),
}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("full", [True, False], ids=["full", "reduced"])
@pytest.mark.parametrize("mat", list(_UV_MATS))
def test_complex_svd_vectors(mat, full, jit):
    # U and V are only defined up to phases (and rotations within repeated or
    # zero singular values), so check shapes, unitarity and reconstruction.
    x = _UV_MATS[mat]()
    m, n = x.shape[-2:]
    k = min(m, n)
    u, s, vh = _on("mps", lambda a: jnp.linalg.svd(a, full_matrices=full), x, jit=jit)
    s_cpu = _on("cpu", lambda a: jnp.linalg.svd(a, compute_uv=False), x, jit=True)
    assert u.dtype == np.complex64 and vh.dtype == np.complex64 and s.dtype == np.float32
    assert u.shape[-2:] == ((m, m) if full else (m, k))
    assert vh.shape[-2:] == ((n, n) if full else (k, n))
    np.testing.assert_allclose(s, s_cpu, rtol=1e-4, atol=2e-4)
    uh = np.conj(np.swapaxes(u, -1, -2))
    np.testing.assert_allclose(uh @ u, np.broadcast_to(np.eye(u.shape[-1]), (*uh.shape[:-1], u.shape[-1])), atol=5e-4)
    vvh = vh @ np.conj(np.swapaxes(vh, -1, -2))
    np.testing.assert_allclose(vvh, np.broadcast_to(np.eye(vh.shape[-2]), vvh.shape), atol=5e-4)
    rec = (u[..., :, :k] * s[..., None, :]) @ vh[..., :k, :]
    np.testing.assert_allclose(rec, x, rtol=1e-4, atol=1e-3)


def test_complex_pinv_matches_cpu():
    x = _cplx((6, 5), 31)
    expected = _on("cpu", jnp.linalg.pinv, x, jit=True)
    actual = _on("mps", jnp.linalg.pinv, x, jit=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
