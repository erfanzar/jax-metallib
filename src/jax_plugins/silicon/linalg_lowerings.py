# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Python-side MPS lowerings for linear-algebra primitives JAX does not lower for MPS.

JAX registers ``eigh_p`` lowerings only for CPU/GPU (LAPACK/cuSOLVER custom
calls) and TPU, so ``jnp.linalg.eigh`` / ``eigvalsh`` failed on MPS with "MLIR
translation rule for primitive 'eigh' not found for platform mps".

The lowering below is a parallel cyclic Jacobi eigensolver written in plain JAX.
Each round rotates n/2 disjoint index pairs at once and applies the combined
rotation as dense matmuls built from one-hot selectors, so the loop body lowers
to matmul and elementwise ops only (no scatter inside the loop). Computation is
in float32. Gradients use JAX's existing ``eigh_p`` JVP rule.

Limitations: real inputs only, no ``subset_by_index``, and cost grows as
O(sweeps * n * n^3), so it is intended for small and medium matrices.
"""

from __future__ import annotations

import numpy as np

_JACOBI_SWEEPS = 12
# Convergence-based early exit (default; JAX_METALLIB_JACOBI_EARLY_EXIT=0
# restores the fixed 12-sweep loop). It cut steady time 20-37% and first-call
# time 2-8x. Its while_loop cond reads the sweep's max rotation sine, a
# reduced carry the lowering does not return, which used to hit an MPSGraph
# segfault (2/12 tail-suite runs); HandleWhileOp's output tie fixes that
# (0/12 afterwards). The tolerance must sit above the float32 noise floor:
# one-sided SVD plateaued at rotation sines of 2e-6..2e-5 from rounding alone.
_JACOBI_TOL = 1e-4


def _early_exit_enabled() -> bool:
    import os

    return os.environ.get("JAX_METALLIB_JACOBI_EARLY_EXIT", "1") != "0"


def _run_sweeps(step, carry, rounds):
    """Run Jacobi sweeps until converged or _JACOBI_SWEEPS (fixed count if opted out).

    step(round_index, carry) -> (carry, rotation_sines).
    """
    import jax.numpy as jnp
    from jax import lax

    if not _early_exit_enabled():

        def fixed_body(i, carry):
            carry, _ = step(i % rounds, carry)
            return carry

        return lax.fori_loop(0, _JACOBI_SWEEPS * rounds, fixed_body, carry)

    def sweep(state):
        k, carry, _ = state

        def body(r, inner):
            carry, maxs = inner
            carry, s = step(r, carry)
            return carry, jnp.maximum(maxs, jnp.max(jnp.abs(s)))

        carry, maxs = lax.fori_loop(0, rounds, body, (carry, jnp.float32(0.0)))
        return k + 1, carry, maxs

    def cond(state):
        k, _, maxs = state
        return (k < _JACOBI_SWEEPS) & (maxs > _JACOBI_TOL)

    _, carry, _ = lax.while_loop(cond, sweep, (jnp.int32(0), carry, jnp.float32(1.0)))
    return carry


def _round_robin_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Round-robin schedule: arrays (rounds, k) of disjoint pairs (p < q)."""
    m = n + (n % 2)
    players = list(range(m))
    ps, qs = [], []
    for _ in range(m - 1):
        rp, rq = [], []
        for i in range(m // 2):
            a, b = players[i], players[m - 1 - i]
            if a < n and b < n:
                rp.append(min(a, b))
                rq.append(max(a, b))
        ps.append(rp)
        qs.append(rq)
        players = [players[0], players[-1], *players[1:-1]]
    return np.asarray(ps, np.int32), np.asarray(qs, np.int32)


def _hermitian_jacobi_eigh(x, *, lower, sort_eigenvalues):
    """Cyclic Jacobi for complex Hermitian (..., n, n) input on float32 (re, im) parts.

    For each pair (p, q) a phase on column q, conj(a_pq) / |a_pq|, makes a_pq
    real; the real Jacobi rotation then zeroes it. The pair unitary is D @ G,
    so A <- (D G)^H A (D G) and V <- V D G. Only real float32 arithmetic is
    used, so no complex matmul reaches MPS.
    """
    import jax.numpy as jnp
    from jax import lax

    n = x.shape[-1]
    real_dtype = jnp.finfo(x.dtype).dtype
    rows = lax.broadcasted_iota(jnp.int32, (n, n), 0)
    cols = lax.broadcasted_iota(jnp.int32, (n, n), 1)
    tri = rows >= cols if lower else rows <= cols
    ar = jnp.real(x).astype(jnp.float32)
    ai = jnp.imag(x).astype(jnp.float32)
    # Mirror the requested triangle as a Hermitian matrix; the imaginary part
    # of the diagonal is ignored, as in LAPACK.
    ar = jnp.where(tri, ar, jnp.swapaxes(ar, -1, -2))
    ai = jnp.where(tri, ai, -jnp.swapaxes(ai, -1, -2))
    ai = jnp.where(rows == cols, 0.0, ai)
    eye = jnp.eye(n, dtype=jnp.float32)
    if n <= 1:
        v = jnp.broadcast_to(eye, ar.shape)
        w = jnp.diagonal(ar, axis1=-2, axis2=-1)
        return lax.complex(v, jnp.zeros_like(v)).astype(x.dtype), w.astype(real_dtype)

    ps_np, qs_np = _round_robin_pairs(n)
    rounds = ps_np.shape[0]
    ps, qs = jnp.asarray(ps_np), jnp.asarray(qs_np)
    iota_n = jnp.arange(n, dtype=jnp.int32)

    def step(r, carry):
        ar, ai, vr, vi = carry
        p = lax.dynamic_index_in_dim(ps, r, keepdims=False)
        q = lax.dynamic_index_in_dim(qs, r, keepdims=False)
        P = (p[:, None] == iota_n[None, :]).astype(jnp.float32)
        Q = (q[:, None] == iota_n[None, :]).astype(jnp.float32)
        d = jnp.diagonal(ar, axis1=-2, axis2=-1)
        app = d @ P.T
        aqq = d @ Q.T
        apq_r = jnp.sum((P @ ar) * Q, axis=-1)
        apq_i = jnp.sum((P @ ai) * Q, axis=-1)
        # |a_pq| as a scaled hypot: squaring a tiny a_pq underflowed, and the
        # resulting "phase" with magnitude > 1 made the update non-unitary
        # (fixed-count sweeps diverged once converged).
        big = jnp.maximum(jnp.abs(apq_r), jnp.abs(apq_i))
        zero = big == 0
        sbig = jnp.where(zero, 1.0, big)
        ur, ui = apq_r / sbig, apq_i / sbig
        nrm = jnp.sqrt(ur * ur + ui * ui)
        nrm = jnp.where(zero, 1.0, nrm)
        mag = big * nrm
        safe = jnp.where(zero, 1.0, mag)
        # Column phases: conj(a_pq) / |a_pq| at every q of this round, 1 elsewhere.
        phr = 1.0 + (jnp.where(zero, 1.0, ur / nrm) - 1.0) @ Q
        phi = jnp.where(zero, 0.0, -ui / nrm) @ Q
        cr, ci = phr[..., None, :], phi[..., None, :]
        mr = ar * cr - ai * ci
        mi = ar * ci + ai * cr
        rr, ri = phr[..., :, None], phi[..., :, None]
        ar = rr * mr + ri * mi
        ai = rr * mi - ri * mr
        # Real Jacobi rotation on the now-real a_pq = |a_pq|.
        theta = (aqq - app) / (2.0 * safe)
        sgn = jnp.where(theta >= 0, 1.0, -1.0)
        t = sgn / (jnp.abs(theta) + jnp.sqrt(theta * theta + 1.0))
        t = jnp.where(zero, 0.0, t)
        c = 1.0 / jnp.sqrt(t * t + 1.0)
        s = t * c
        cm1 = (c - 1.0)[..., :, None]
        sk = s[..., :, None]
        G = eye + P.T @ (cm1 * P) + Q.T @ (cm1 * Q) + P.T @ (sk * Q) - Q.T @ (sk * P)
        Gt = jnp.swapaxes(G, -1, -2)
        ar = Gt @ ar @ G
        ai = Gt @ ai @ G
        wr = vr * cr - vi * ci
        wi = vr * ci + vi * cr
        return (ar, ai, wr @ G, wi @ G), s

    v0 = jnp.broadcast_to(eye, ar.shape)
    ar, ai, vr, vi = _run_sweeps(step, (ar, ai, v0, jnp.zeros_like(v0)), rounds)
    w = jnp.diagonal(ar, axis1=-2, axis2=-1)
    if sort_eigenvalues:
        order = jnp.argsort(w, axis=-1)
        w = jnp.take_along_axis(w, order, axis=-1)
        vr = jnp.take_along_axis(vr, order[..., None, :], axis=-1)
        vi = jnp.take_along_axis(vi, order[..., None, :], axis=-1)
    return lax.complex(vr, vi).astype(x.dtype), w.astype(real_dtype)


def _jacobi_eigh(x, *, lower, sort_eigenvalues, subset_by_index, algorithm):
    import jax.numpy as jnp
    from jax import lax

    del algorithm
    n = x.shape[-1]
    if subset_by_index is not None and tuple(subset_by_index) != (0, n):
        raise NotImplementedError("eigh with subset_by_index is not supported on MPS")
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        return _hermitian_jacobi_eigh(x, lower=lower, sort_eigenvalues=sort_eigenvalues)

    out_dtype = x.dtype
    a = x.astype(jnp.float32)
    eye = jnp.eye(n, dtype=jnp.float32)
    if n <= 1:
        v = jnp.broadcast_to(eye, a.shape)
        return v.astype(out_dtype), jnp.diagonal(a, axis1=-2, axis2=-1).astype(out_dtype)

    # Read only the requested triangle and mirror it into the other one.
    rows = lax.broadcasted_iota(jnp.int32, (n, n), 0)
    cols = lax.broadcasted_iota(jnp.int32, (n, n), 1)
    tri = rows >= cols if lower else rows <= cols
    a = jnp.where(tri, a, jnp.swapaxes(a, -1, -2))

    ps_np, qs_np = _round_robin_pairs(n)
    rounds = ps_np.shape[0]
    ps, qs = jnp.asarray(ps_np), jnp.asarray(qs_np)
    iota_n = jnp.arange(n, dtype=jnp.int32)

    def step(r, carry):
        a, v = carry
        p = lax.dynamic_index_in_dim(ps, r, keepdims=False)
        q = lax.dynamic_index_in_dim(qs, r, keepdims=False)
        P = (p[:, None] == iota_n[None, :]).astype(jnp.float32)  # (k, n) one-hot rows
        Q = (q[:, None] == iota_n[None, :]).astype(jnp.float32)
        d = jnp.diagonal(a, axis1=-2, axis2=-1)
        app = d @ P.T
        aqq = d @ Q.T
        apq = jnp.sum((P @ a) * Q, axis=-1)
        zero = apq == 0
        theta = (aqq - app) / (2.0 * jnp.where(zero, 1.0, apq))
        sgn = jnp.where(theta >= 0, 1.0, -1.0)
        t = sgn / (jnp.abs(theta) + jnp.sqrt(theta * theta + 1.0))
        t = jnp.where(zero, 0.0, t)
        c = 1.0 / jnp.sqrt(t * t + 1.0)
        s = t * c
        cm1 = (c - 1.0)[..., :, None]
        sk = s[..., :, None]
        G = eye + P.T @ (cm1 * P) + Q.T @ (cm1 * Q) + P.T @ (sk * Q) - Q.T @ (sk * P)
        a = jnp.swapaxes(G, -1, -2) @ a @ G
        v = v @ G
        return (a, v), s

    v0 = jnp.broadcast_to(eye, a.shape)
    a, v = _run_sweeps(step, (a, v0), rounds)
    w = jnp.diagonal(a, axis1=-2, axis2=-1)
    if sort_eigenvalues:
        order = jnp.argsort(w, axis=-1)
        w = jnp.take_along_axis(w, order, axis=-1)
        v = jnp.take_along_axis(v, order[..., None, :], axis=-1)
    return v.astype(out_dtype), w.astype(out_dtype)


def _one_sided_jacobi(a):
    """Hestenes one-sided Jacobi on a tall (..., m, n) float32 matrix.

    Returns (s, u_partial, v): column norms (unsorted), the normalised
    columns (zero where s == 0) and the accumulated right rotations (n x n).
    """
    import jax.numpy as jnp
    from jax import lax

    n = a.shape[-1]
    eye = jnp.eye(n, dtype=jnp.float32)
    v0 = jnp.broadcast_to(eye, (*a.shape[:-2], n, n))
    if n > 1:
        ps_np, qs_np = _round_robin_pairs(n)
        rounds = ps_np.shape[0]
        ps, qs = jnp.asarray(ps_np), jnp.asarray(qs_np)
        iota_n = jnp.arange(n, dtype=jnp.int32)

        def step(r, carry):
            a, v = carry
            p = lax.dynamic_index_in_dim(ps, r, keepdims=False)
            q = lax.dynamic_index_in_dim(qs, r, keepdims=False)
            P = (p[:, None] == iota_n[None, :]).astype(jnp.float32)
            Q = (q[:, None] == iota_n[None, :]).astype(jnp.float32)
            ap = a @ P.T  # (..., m, k) columns p
            aq = a @ Q.T
            alpha = jnp.sum(ap * ap, axis=-2)
            beta = jnp.sum(aq * aq, axis=-2)
            gamma = jnp.sum(ap * aq, axis=-2)
            zero = gamma == 0
            zeta = (beta - alpha) / (2.0 * jnp.where(zero, 1.0, gamma))
            sgn = jnp.where(zeta >= 0, 1.0, -1.0)
            t = sgn / (jnp.abs(zeta) + jnp.sqrt(zeta * zeta + 1.0))
            t = jnp.where(zero, 0.0, t)
            c = 1.0 / jnp.sqrt(t * t + 1.0)
            s = t * c
            cm1 = (c - 1.0)[..., :, None]
            sk = s[..., :, None]
            G = eye + P.T @ (cm1 * P) + Q.T @ (cm1 * Q) + P.T @ (sk * Q) - Q.T @ (sk * P)
            return (a @ G, v @ G), s

        a, v0 = _run_sweeps(step, (a, v0), rounds)
    s = jnp.sqrt(jnp.sum(a * a, axis=-2))
    u = a / jnp.where(s > 0, s, 1.0)[..., None, :]
    return s, u, v0


def _orthonormal_completion(u_partial, *, complete):
    """Orthonormalise u_partial (..., m, k) via QR, keeping its column signs."""
    import jax.numpy as jnp

    k = u_partial.shape[-1]
    q, r = jnp.linalg.qr(u_partial, mode="complete" if complete else "reduced")
    d = jnp.diagonal(r, axis1=-2, axis2=-1)
    sign = jnp.where(d < 0, -1.0, 1.0).astype(q.dtype)
    head = q[..., :, :k] * sign[..., None, :]
    return jnp.concatenate([head, q[..., :, k:]], axis=-1) if complete else head


def _complex_one_sided_jacobi(ar, ai):
    """Hestenes one-sided Jacobi on a tall complex (..., m, n) matrix given as float32 (re, im).

    Per pair a phase conj(g)/|g| on column q, with g = a_p^H a_q, makes g real;
    the real rotation then orthogonalises the pair. Returns (s, ar, ai, vr, vi):
    column norms (unsorted), the rotated columns and the accumulated unitary V.
    """
    import jax.numpy as jnp
    from jax import lax

    n = ar.shape[-1]
    eye = jnp.eye(n, dtype=jnp.float32)
    vr = jnp.broadcast_to(eye, (*ar.shape[:-2], n, n))
    vi = jnp.zeros_like(vr)
    if n > 1:
        ps_np, qs_np = _round_robin_pairs(n)
        rounds = ps_np.shape[0]
        ps, qs = jnp.asarray(ps_np), jnp.asarray(qs_np)
        iota_n = jnp.arange(n, dtype=jnp.int32)

        def step(r, carry):
            ar, ai, vr, vi = carry
            p = lax.dynamic_index_in_dim(ps, r, keepdims=False)
            q = lax.dynamic_index_in_dim(qs, r, keepdims=False)
            P = (p[:, None] == iota_n[None, :]).astype(jnp.float32)
            Q = (q[:, None] == iota_n[None, :]).astype(jnp.float32)
            apr, api = ar @ P.T, ai @ P.T
            aqr, aqi = ar @ Q.T, ai @ Q.T
            alpha = jnp.sum(apr * apr + api * api, axis=-2)
            beta = jnp.sum(aqr * aqr + aqi * aqi, axis=-2)
            gr = jnp.sum(apr * aqr + api * aqi, axis=-2)
            gi = jnp.sum(apr * aqi - api * aqr, axis=-2)
            # |g| as a scaled hypot (squaring a tiny g underflows; see eigh).
            big = jnp.maximum(jnp.abs(gr), jnp.abs(gi))
            zero = big == 0
            sbig = jnp.where(zero, 1.0, big)
            nr, ni = gr / sbig, gi / sbig
            nrm = jnp.where(zero, 1.0, jnp.sqrt(nr * nr + ni * ni))
            safe = jnp.where(zero, 1.0, big * nrm)
            phr = 1.0 + (jnp.where(zero, 1.0, nr / nrm) - 1.0) @ Q
            phi = jnp.where(zero, 0.0, -ni / nrm) @ Q
            cr, ci = phr[..., None, :], phi[..., None, :]
            ar, ai = ar * cr - ai * ci, ar * ci + ai * cr
            vr, vi = vr * cr - vi * ci, vr * ci + vi * cr
            zeta = (beta - alpha) / (2.0 * safe)
            sgn = jnp.where(zeta >= 0, 1.0, -1.0)
            t = sgn / (jnp.abs(zeta) + jnp.sqrt(zeta * zeta + 1.0))
            t = jnp.where(zero, 0.0, t)
            c = 1.0 / jnp.sqrt(t * t + 1.0)
            s = t * c
            cm1 = (c - 1.0)[..., :, None]
            sk = s[..., :, None]
            G = eye + P.T @ (cm1 * P) + Q.T @ (cm1 * Q) + P.T @ (sk * Q) - Q.T @ (sk * P)
            return (ar @ G, ai @ G, vr @ G, vi @ G), s

        ar, ai, vr, vi = _run_sweeps(step, (ar, ai, vr, vi), rounds)
    s = jnp.sqrt(jnp.sum(ar * ar + ai * ai, axis=-2))
    return s, ar, ai, vr, vi


def _complex_orthonormal_completion(ur, ui, keep, width):
    """Fill columns of (ur, ui) (..., m, k) where keep is False, and pad to width.

    Kept columns (a prefix) are orthonormal. The rest come from eigenvectors of
    the Hermitian projector I - U U^H with eigenvalue 1 (its orthogonal
    complement), computed with the complex Jacobi eigh; no complex QR needed.
    """
    import jax.numpy as jnp
    from jax import lax

    m, k = ur.shape[-2:]
    if width > k:
        pad = [(0, 0)] * (ur.ndim - 1) + [(0, width - k)]
        ur, ui = jnp.pad(ur, pad), jnp.pad(ui, pad)
        keep = jnp.pad(keep, [(0, 0)] * (keep.ndim - 1) + [(0, width - k)])
    urt, uit = jnp.swapaxes(ur, -1, -2), jnp.swapaxes(ui, -1, -2)
    pr = jnp.eye(m, dtype=jnp.float32) - (ur @ urt + ui @ uit)
    pi = ur @ uit - ui @ urt
    w, _ = _hermitian_jacobi_eigh(lax.complex(pr, pi), lower=True, sort_eigenvalues=True)
    wr, wi = jnp.real(w).astype(jnp.float32), jnp.imag(w).astype(jnp.float32)
    # Eigenvalues ascend: 0 on span(U), 1 on the complement at the end.
    r = jnp.sum(keep.astype(jnp.int32), axis=-1, keepdims=True)
    j = jnp.arange(width, dtype=jnp.int32)
    src = jnp.clip(m - 1 - (j - r), 0, m - 1)
    idx = jnp.broadcast_to(src[..., None, :], (*wr.shape[:-1], width))
    cr = jnp.take_along_axis(wr, idx, axis=-1)
    ci = jnp.take_along_axis(wi, idx, axis=-1)
    use = (j < r)[..., None, :]
    return jnp.where(use, ur, cr), jnp.where(use, ui, ci)


def _complex_jacobi_svd(x, *, full_matrices, subset_by_index):
    import jax.numpy as jnp
    from jax import lax

    m, n = x.shape[-2:]
    rank = min(m, n)
    lo, hi = (0, rank) if subset_by_index is None else tuple(subset_by_index)
    real_dtype = jnp.finfo(x.dtype).dtype
    br = jnp.real(x).astype(jnp.float32)
    bi = jnp.imag(x).astype(jnp.float32)
    wide = m < n
    if wide:  # a^H = U' S V'^H, so a = V' S U'^H
        br, bi = jnp.swapaxes(br, -1, -2), -jnp.swapaxes(bi, -1, -2)
    tall_m, k = br.shape[-2:]
    s, ar, ai, vr, vi = _complex_one_sided_jacobi(br, bi)
    order = jnp.argsort(-s, axis=-1)
    s = jnp.take_along_axis(s, order, axis=-1)
    cols = order[..., None, :]
    ar, ai = jnp.take_along_axis(ar, cols, axis=-1), jnp.take_along_axis(ai, cols, axis=-1)
    vr, vi = jnp.take_along_axis(vr, cols, axis=-1), jnp.take_along_axis(vi, cols, axis=-1)
    # Columns with negligible singular values are rounding noise; replace them
    # with an orthonormal complement instead of normalising the noise.
    tol = jnp.max(s, axis=-1, keepdims=True) * (32 * max(m, n) * float(jnp.finfo(jnp.float32).eps))
    keep = s > tol
    denom = jnp.where(keep, s, 1.0)[..., None, :]
    ur = jnp.where(keep[..., None, :], ar / denom, 0.0)
    ui = jnp.where(keep[..., None, :], ai / denom, 0.0)
    ur, ui = _complex_orthonormal_completion(ur, ui, keep, tall_m if full_matrices else k)
    if wide:
        u = lax.complex(vr, vi)
        vh = lax.complex(jnp.swapaxes(ur, -1, -2), -jnp.swapaxes(ui, -1, -2))
    else:
        u = lax.complex(ur, ui)
        vh = lax.complex(jnp.swapaxes(vr, -1, -2), -jnp.swapaxes(vi, -1, -2))
    if not full_matrices:
        u, vh = u[..., :, lo:hi], vh[..., lo:hi, :]
    return s[..., lo:hi].astype(real_dtype), u.astype(x.dtype), vh.astype(x.dtype)


def _jacobi_svd(x, *, full_matrices, compute_uv, subset_by_index, algorithm=None):
    import jax.numpy as jnp

    del algorithm
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        if compute_uv:
            return _complex_jacobi_svd(x, full_matrices=full_matrices, subset_by_index=subset_by_index)
        # B + iC has the same singular values as the real embedding
        # [[B, -C], [C, B]], each repeated twice.
        m, n = x.shape[-2:]
        rank = min(m, n)
        lo, hi = (0, rank) if subset_by_index is None else tuple(subset_by_index)
        br = jnp.real(x).astype(jnp.float32)
        bi = jnp.imag(x).astype(jnp.float32)
        e = jnp.concatenate([jnp.concatenate([br, -bi], axis=-1), jnp.concatenate([bi, br], axis=-1)], axis=-2)
        if m < n:
            e = jnp.swapaxes(e, -1, -2)
        s, _, _ = _one_sided_jacobi(e)
        s = -jnp.sort(-s, axis=-1)
        return (s[..., 0::2][..., lo:hi].astype(jnp.finfo(x.dtype).dtype),)
    m, n = x.shape[-2:]
    rank = min(m, n)
    lo, hi = (0, rank) if subset_by_index is None else tuple(subset_by_index)
    out_dtype = x.dtype
    a = x.astype(jnp.float32)
    wide = m < n
    if wide:
        a = jnp.swapaxes(a, -1, -2)  # tall: a^T = U' S V'^T, so a = V' S U'^T
    s, u_part, v = _one_sided_jacobi(a)
    order = jnp.argsort(-s, axis=-1)
    s = jnp.take_along_axis(s, order, axis=-1)
    s_out = s[..., lo:hi].astype(out_dtype)
    if not compute_uv:
        return (s_out,)
    u_part = jnp.take_along_axis(u_part, order[..., None, :], axis=-1)
    v = jnp.take_along_axis(v, order[..., None, :], axis=-1)
    tall_u = _orthonormal_completion(u_part, complete=full_matrices)
    if wide:
        u, vt = v, jnp.swapaxes(tall_u, -1, -2)
    else:
        u, vt = tall_u, jnp.swapaxes(v, -1, -2)
    if not full_matrices:
        u, vt = u[..., :, lo:hi], vt[..., lo:hi, :]
    return s_out, u.astype(out_dtype), vt.astype(out_dtype)


def _swap(t):
    import jax.numpy as jnp

    return jnp.swapaxes(t, -1, -2)


def _lower_substitution(mr, mi, br, bi):
    """Solve M X = B for lower-triangular complex M given as float32 (re, im) parts.

    X starts at zero and row i is written at step i, so M[i] @ X is exactly
    sum_{j<i} M[i, j] X[j] (the diagonal term multiplies a still-zero row).
    """
    import jax.numpy as jnp
    from jax import lax

    n = mr.shape[-1]
    row_axis = mr.ndim - 2
    dr = jnp.diagonal(mr, axis1=-2, axis2=-1)
    di = jnp.diagonal(mi, axis1=-2, axis2=-1)
    den = dr * dr + di * di

    def body(i, x):
        xr, xi = x
        rr = lax.dynamic_index_in_dim(mr, i, axis=row_axis, keepdims=True)
        ri = lax.dynamic_index_in_dim(mi, i, axis=row_axis, keepdims=True)
        tr = lax.dynamic_index_in_dim(br, i, axis=row_axis, keepdims=True) - (rr @ xr - ri @ xi)
        ti = lax.dynamic_index_in_dim(bi, i, axis=row_axis, keepdims=True) - (rr @ xi + ri @ xr)
        d_r = lax.dynamic_index_in_dim(dr, i, axis=dr.ndim - 1, keepdims=True)[..., None]
        d_i = lax.dynamic_index_in_dim(di, i, axis=di.ndim - 1, keepdims=True)[..., None]
        dd = lax.dynamic_index_in_dim(den, i, axis=den.ndim - 1, keepdims=True)[..., None]
        yr = (tr * d_r + ti * d_i) / dd
        yi = (ti * d_r - tr * d_i) / dd
        xr = lax.dynamic_update_index_in_dim(xr, yr, i, axis=row_axis)
        xi = lax.dynamic_update_index_in_dim(xi, yi, i, axis=row_axis)
        return xr, xi

    return lax.fori_loop(0, n, body, (jnp.zeros_like(br), jnp.zeros_like(bi)))


def _complex_triangular_solve(a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
    """triangular_solve for complex inputs on float32 (re, im) parts.

    The native MPS solver is float-only and returned nil for complex64, which
    broke complex solve, inv, cho_solve and expm. Every variant is reduced to a
    left-side lower solve (conjugate, transpose, right side via transposition,
    upper via row/column reversal). Only the requested triangle is read.
    """
    import jax.numpy as jnp
    from jax import lax

    f32 = jnp.float32
    n = a.shape[-1]
    tri = jnp.tril if lower else jnp.triu
    ar, ai = tri(jnp.real(a).astype(f32)), tri(jnp.imag(a).astype(f32))
    if unit_diagonal:
        eye = jnp.eye(n, dtype=f32)
        ar = ar * (1.0 - eye) + eye
        ai = ai * (1.0 - eye)
    if conjugate_a:
        ai = -ai
    low = lower
    if transpose_a:
        ar, ai, low = _swap(ar), _swap(ai), not low
    br, bi = jnp.real(b).astype(f32), jnp.imag(b).astype(f32)
    if not left_side:  # X M = B  <=>  M^T X^T = B^T
        ar, ai, low = _swap(ar), _swap(ai), not low
        br, bi = _swap(br), _swap(bi)
    if not low:  # J M J is lower: solve (J M J)(J X) = J B
        ar, ai = jnp.flip(ar, (-2, -1)), jnp.flip(ai, (-2, -1))
        br, bi = jnp.flip(br, -2), jnp.flip(bi, -2)
    xr, xi = _lower_substitution(ar, ai, br, bi)
    if not low:
        xr, xi = jnp.flip(xr, -2), jnp.flip(xi, -2)
    if not left_side:
        xr, xi = _swap(xr), _swap(xi)
    return lax.complex(xr, xi).astype(b.dtype)


def _complex_cholesky(x):
    """Lower Cholesky of a complex Hermitian matrix on float32 (re, im) parts.

    Column j: v = A[:, j] - L conj(L[j, :])^T (columns >= j of L are still zero),
    L[j, j] = sqrt(re v[j]), L[i > j, j] = v[i] / L[j, j]. Only the lower
    triangle is read and the diagonal's imaginary part is ignored.
    """
    import jax.numpy as jnp
    from jax import lax

    n = x.shape[-1]
    ar = jnp.real(x).astype(jnp.float32)
    ai = jnp.imag(x).astype(jnp.float32)
    row_axis = ar.ndim - 2
    col_axis = ar.ndim - 1
    rows = jnp.arange(n)[:, None]

    def body(j, carry):
        lr, li = carry
        rjr = _swap(lax.dynamic_index_in_dim(lr, j, axis=row_axis, keepdims=True))
        rji = _swap(lax.dynamic_index_in_dim(li, j, axis=row_axis, keepdims=True))
        vr = lax.dynamic_index_in_dim(ar, j, axis=col_axis, keepdims=True) - (lr @ rjr + li @ rji)
        vi = lax.dynamic_index_in_dim(ai, j, axis=col_axis, keepdims=True) - (li @ rjr - lr @ rji)
        d = jnp.sqrt(lax.dynamic_index_in_dim(vr, j, axis=row_axis, keepdims=True))
        below, diag = rows > j, rows == j
        cr = jnp.where(below, vr / d, jnp.where(diag, d, 0.0))
        ci = jnp.where(below, vi / d, 0.0)
        lr = lax.dynamic_update_index_in_dim(lr, cr, j, axis=col_axis)
        li = lax.dynamic_update_index_in_dim(li, ci, j, axis=col_axis)
        return lr, li

    lr, li = lax.fori_loop(0, n, body, (jnp.zeros_like(ar), jnp.zeros_like(ai)))
    return lax.complex(lr, li).astype(x.dtype)


def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def _complex_geqrf(x):
    """Unblocked Householder QR of complex (..., m, n) input (LAPACK zgeqr2 / zlarfg).

    Works on float32 (re, im) parts. Step i: alpha = A[i, i], beta =
    -sign(re alpha) * ||A[i:, i]|| (real), tau = (beta - alpha) / beta,
    v = [1, A[i+1:, i] / (alpha - beta)], then H^H = I - conj(tau) v v^H is
    applied to the trailing columns. Returns the packed matrix (R on and
    above the diagonal, reflector tails below) and taus, as LAPACK does.
    """
    import jax.numpy as jnp
    from jax import lax

    m, n = x.shape[-2:]
    k = min(m, n)
    ar = jnp.real(x).astype(jnp.float32)
    ai = jnp.imag(x).astype(jnp.float32)
    col_axis = ar.ndim - 1
    rows = jnp.arange(m)
    cols = jnp.arange(n)
    taur = jnp.zeros((*ar.shape[:-2], k), jnp.float32)
    taui = jnp.zeros_like(taur)

    def body(i, carry):
        ar, ai, taur, taui = carry
        xr = lax.dynamic_index_in_dim(ar, i, axis=col_axis, keepdims=False)
        xi = lax.dynamic_index_in_dim(ai, i, axis=col_axis, keepdims=False)
        below = rows > i
        at = rows == i
        alr = jnp.sum(jnp.where(at, xr, 0.0), axis=-1)
        ali = jnp.sum(jnp.where(at, xi, 0.0), axis=-1)
        tr = jnp.where(below, xr, 0.0)
        ti = jnp.where(below, xi, 0.0)
        xnorm2 = jnp.sum(tr * tr + ti * ti, axis=-1)
        trivial = (xnorm2 == 0) & (ali == 0)
        norm = jnp.sqrt(alr * alr + ali * ali + xnorm2)
        beta = jnp.where(alr >= 0, -norm, norm)
        beta = jnp.where(trivial, alr, beta)
        sbeta = jnp.where(trivial, 1.0, beta)
        t_r = jnp.where(trivial, 0.0, (beta - alr) / sbeta)
        t_i = jnp.where(trivial, 0.0, -ali / sbeta)
        # v tail = tail / (alpha - beta)
        dr = jnp.where(trivial, 1.0, alr - beta)
        di = jnp.where(trivial, 0.0, ali)
        den = dr * dr + di * di
        vtr = (tr * dr[..., None] + ti * di[..., None]) / den[..., None]
        vti = (ti * dr[..., None] - tr * di[..., None]) / den[..., None]
        vr = jnp.where(at, 1.0, vtr)
        vi = jnp.where(at, 0.0, vti)
        # w_j = sum_r conj(v_r) A_rj, then A -= conj(tau) v w on columns > i.
        wr = jnp.einsum("...r,...rj->...j", vr, ar) + jnp.einsum("...r,...rj->...j", vi, ai)
        wi = jnp.einsum("...r,...rj->...j", vr, ai) - jnp.einsum("...r,...rj->...j", vi, ar)
        ur, ui = _cmul(vr[..., :, None], vi[..., :, None], wr[..., None, :], wi[..., None, :])
        ctr, cti = t_r[..., None, None], -t_i[..., None, None]
        dr2, di2 = _cmul(ctr, cti, ur, ui)
        right = (cols > i)[None, :]
        ar = jnp.where(right, ar - dr2, ar)
        ai = jnp.where(right, ai - di2, ai)
        # Column i: rows < i unchanged, beta on the diagonal, v tail below.
        newr = jnp.where(at, beta[..., None], jnp.where(below, vtr, xr))
        newi = jnp.where(at, 0.0, jnp.where(below, vti, xi))
        ar = lax.dynamic_update_index_in_dim(ar, newr[..., :, None], i, axis=col_axis)
        ai = lax.dynamic_update_index_in_dim(ai, newi[..., :, None], i, axis=col_axis)
        taur = lax.dynamic_update_index_in_dim(taur, t_r[..., None], i, axis=taur.ndim - 1)
        taui = lax.dynamic_update_index_in_dim(taui, t_i[..., None], i, axis=taui.ndim - 1)
        return ar, ai, taur, taui

    ar, ai, taur, taui = lax.fori_loop(0, k, body, (ar, ai, taur, taui))
    return [lax.complex(ar, ai).astype(x.dtype), lax.complex(taur, taui).astype(x.dtype)]


def _complex_householder_product(a, taus):
    """Q = H_0 ... H_{k-1} applied to I[:, :n] (LAPACK zung2r), H_i = I - tau_i v_i v_i^H."""
    import jax.numpy as jnp
    from jax import lax

    m, n = a.shape[-2:]
    k = taus.shape[-1]
    vr_all = jnp.real(a).astype(jnp.float32)
    vi_all = jnp.imag(a).astype(jnp.float32)
    tr_all = jnp.real(taus).astype(jnp.float32)
    ti_all = jnp.imag(taus).astype(jnp.float32)
    col_axis = vr_all.ndim - 1
    rows = jnp.arange(m)
    qr = jnp.broadcast_to(jnp.eye(m, n, dtype=jnp.float32), vr_all.shape)
    qi = jnp.zeros_like(qr)

    def body(s, carry):
        qr, qi = carry
        i = k - 1 - s
        xr = lax.dynamic_index_in_dim(vr_all, i, axis=col_axis, keepdims=False)
        xi = lax.dynamic_index_in_dim(vi_all, i, axis=col_axis, keepdims=False)
        vr = jnp.where(rows == i, 1.0, jnp.where(rows > i, xr, 0.0))
        vi = jnp.where(rows > i, xi, 0.0)
        t_r = lax.dynamic_index_in_dim(tr_all, i, axis=tr_all.ndim - 1, keepdims=True)[..., None]
        t_i = lax.dynamic_index_in_dim(ti_all, i, axis=ti_all.ndim - 1, keepdims=True)[..., None]
        wr = jnp.einsum("...r,...rj->...j", vr, qr) + jnp.einsum("...r,...rj->...j", vi, qi)
        wi = jnp.einsum("...r,...rj->...j", vr, qi) - jnp.einsum("...r,...rj->...j", vi, qr)
        ur, ui = _cmul(vr[..., :, None], vi[..., :, None], wr[..., None, :], wi[..., None, :])
        dr, di = _cmul(t_r, t_i, ur, ui)
        return qr - dr, qi - di

    qr, qi = lax.fori_loop(0, k, body, (qr, qi))
    return lax.complex(qr, qi).astype(a.dtype)


def _hh_vector(xr, xi, p, rows):
    """LAPACK zlarfg reflector for column (xr, xi) with pivot row p (rows < p ignored).

    Returns v (v[p] = 1, zero above p), tau, and the stored tail v[p+1:] (zero
    elsewhere). beta = -sign(re alpha) * norm is real; tau = 0 when the column
    below p is zero and alpha is real.
    """
    import jax.numpy as jnp

    at = rows == p
    below = rows > p
    alr = jnp.sum(jnp.where(at, xr, 0.0), axis=-1)
    ali = jnp.sum(jnp.where(at, xi, 0.0), axis=-1)
    tr = jnp.where(below, xr, 0.0)
    ti = jnp.where(below, xi, 0.0)
    xnorm2 = jnp.sum(tr * tr + ti * ti, axis=-1)
    trivial = (xnorm2 == 0) & (ali == 0)
    norm = jnp.sqrt(alr * alr + ali * ali + xnorm2)
    beta = jnp.where(trivial, alr, jnp.where(alr >= 0, -norm, norm))
    sbeta = jnp.where(trivial, 1.0, beta)
    t_r = jnp.where(trivial, 0.0, (beta - alr) / sbeta)
    t_i = jnp.where(trivial, 0.0, -ali / sbeta)
    dr = jnp.where(trivial, 1.0, alr - beta)
    di = jnp.where(trivial, 0.0, ali)
    den = (dr * dr + di * di)[..., None]
    vtr = (tr * dr[..., None] + ti * di[..., None]) / den
    vti = (ti * dr[..., None] - tr * di[..., None]) / den
    return jnp.where(at, 1.0, vtr), jnp.where(at, 0.0, vti), t_r, t_i, vtr, vti


def _similarity(wr, wi, vr, vi, t_r, t_i):
    """W <- H^H W H with H = I - tau v v^H, on float32 (re, im) parts."""
    import jax.numpy as jnp

    def mv(m, v):
        return jnp.einsum("...rc,...c->...r", m, v)

    def vm(v, m):
        return jnp.einsum("...r,...rc->...c", v, m)

    # W H = W - (tau W v) v^H
    pr, pi = mv(wr, vr) - mv(wi, vi), mv(wr, vi) + mv(wi, vr)
    ar, ai = _cmul(t_r[..., None], t_i[..., None], pr, pi)
    wr = wr - (ar[..., :, None] * vr[..., None, :] + ai[..., :, None] * vi[..., None, :])
    wi = wi - (ai[..., :, None] * vr[..., None, :] - ar[..., :, None] * vi[..., None, :])
    # H^H W = W - (conj(tau) v) (v^H W)
    qr, qi = vm(vr, wr) + vm(vi, wi), vm(vr, wi) - vm(vi, wr)
    cr, ci = _cmul(t_r[..., None], -t_i[..., None], vr, vi)
    wr = wr - (cr[..., :, None] * qr[..., None, :] - ci[..., :, None] * qi[..., None, :])
    wi = wi - (cr[..., :, None] * qi[..., None, :] + ci[..., :, None] * qr[..., None, :])
    return wr, wi


def _householder_reduce(wr, wi):
    """Reduce column i below row i+1 for i < n-1 by Householder similarities.

    Returns the reduced (re, im), the reflector tails packed below the first
    subdiagonal, and taus (..., n-1).
    """
    import jax.numpy as jnp
    from jax import lax

    n = wr.shape[-1]
    k = n - 1
    rows = jnp.arange(n)
    col_axis = wr.ndim - 1
    tlr, tli = jnp.zeros_like(wr), jnp.zeros_like(wi)
    tar = jnp.zeros((*wr.shape[:-2], k), jnp.float32)
    tai = jnp.zeros_like(tar)

    def body(i, carry):
        wr, wi, tlr, tli, tar, tai = carry
        xr = lax.dynamic_index_in_dim(wr, i, axis=col_axis, keepdims=False)
        xi = lax.dynamic_index_in_dim(wi, i, axis=col_axis, keepdims=False)
        vr, vi, t_r, t_i, vtr, vti = _hh_vector(xr, xi, i + 1, rows)
        wr, wi = _similarity(wr, wi, vr, vi, t_r, t_i)
        tlr = lax.dynamic_update_index_in_dim(tlr, vtr[..., :, None], i, axis=col_axis)
        tli = lax.dynamic_update_index_in_dim(tli, vti[..., :, None], i, axis=col_axis)
        tar = lax.dynamic_update_index_in_dim(tar, t_r[..., None], i, axis=tar.ndim - 1)
        tai = lax.dynamic_update_index_in_dim(tai, t_i[..., None], i, axis=tai.ndim - 1)
        return wr, wi, tlr, tli, tar, tai

    carry = (wr, wi, tlr, tli, tar, tai)
    if k > 0:
        carry = lax.fori_loop(0, k, body, carry)
    return carry


def _pack(dtype, re, im):
    import jax.numpy as jnp
    from jax import lax

    if jnp.issubdtype(dtype, jnp.complexfloating):
        return lax.complex(re, im).astype(dtype)
    return re.astype(dtype)


def _hessenberg(a):
    """Upper Hessenberg reduction in LAPACK gehrd packed form (real and complex)."""
    import jax.numpy as jnp

    n = a.shape[-1]
    wr = jnp.real(a).astype(jnp.float32)
    wi = jnp.imag(a).astype(jnp.float32)
    wr, wi, tlr, tli, tar, tai = _householder_reduce(wr, wi)
    rows = jnp.arange(n)
    below = rows[:, None] > rows[None, :] + 1
    pr = jnp.where(below, tlr, wr)
    pi = jnp.where(below, tli, wi)
    return [_pack(a.dtype, pr, pi), _pack(a.dtype, tar, tai)]


def _tridiagonal(a, *, lower):
    """Hermitian tridiagonal reduction in LAPACK sytrd/hetrd packed form.

    Only the requested triangle is read; the other triangle of the packed
    output is the input's, unchanged. The upper variant is the lower one on
    J A J (J = reversal) with every output flipped back.
    """
    import jax.numpy as jnp

    if not lower:
        arr, d, e, taus = _tridiagonal(jnp.flip(a, (-2, -1)), lower=True)
        return [jnp.flip(arr, (-2, -1)), jnp.flip(d, -1), jnp.flip(e, -1), jnp.flip(taus, -1)]
    n = a.shape[-1]
    ar = jnp.real(a).astype(jnp.float32)
    ai = jnp.imag(a).astype(jnp.float32)
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    wr = jnp.where(rows >= cols, ar, _swap(ar))
    wi = jnp.where(rows > cols, ai, jnp.where(rows < cols, -_swap(ai), 0.0))
    wr, wi, tlr, tli, tar, tai = _householder_reduce(wr, wi)
    real_dtype = jnp.finfo(a.dtype).dtype
    d = jnp.diagonal(wr, axis1=-2, axis2=-1).astype(real_dtype)
    e = jnp.diagonal(wr, offset=-1, axis1=-2, axis2=-1).astype(real_dtype)
    below = rows > cols + 1
    pr = jnp.where(below, tlr, jnp.where(rows >= cols, wr, ar))
    pi = jnp.where(below, tli, jnp.where(rows >= cols, 0.0, ai))
    return [_pack(a.dtype, pr, pi), d, e, _pack(a.dtype, tar, tai)]


_SWEEP_COUNTED_MIN_N = 64


def _sweep_loop(cond, body, state, maxit, n):
    """while_loop(cond, body, state), capped at maxit iterations.

    For n >= 64 the shifted-QR sweep inside lax.while_loop (MPSGraph
    whileWithInitialInputs) aborted the process: MPSGPURegionFormation failed
    with 'operand #0 does not dominate this use'. The same body compiles inside
    a counted loop (forLoopWithLowerBound), so large n uses a fori_loop over
    the cap with the sweep guarded by lax.cond. Two-branch stablehlo.case
    lowers to an MPSGraph if, so converged iterations skip the sweep (with
    JAX_METALLIB_DISABLE_CASE_GRAPH_IF=1 both branches run and every capped
    iteration pays a full sweep); smaller n keeps the while_loop unchanged.
    """
    from jax import lax

    if n < _SWEEP_COUNTED_MIN_N:
        return lax.while_loop(cond, body, state)

    def step(_, s):
        return lax.cond(cond(s), body, lambda c: c, s)

    return lax.fori_loop(0, maxit, step, state)


def _complex_schur(x, *, compute_schur_vectors, sort_eig_vals=False, select_callable=None):
    """Complex Schur decomposition A = Z T Z^H on float32 (re, im) parts.

    Hessenberg reduction (accumulating Z), then explicit Wilkinson-shifted QR
    sweeps with complex Givens rotations on the active leading block,
    deflating the bottom subdiagonal entry when it is negligible. An
    exceptional shift is used every 10 sweeps without deflation; the sweep
    count is capped at 40 n. hi-dependent quantities use masks, so batches need
    no gathers. Real input takes the real quasi-triangular form (_real_schur).
    """
    import jax.numpy as jnp
    from jax import lax

    del select_callable
    if sort_eig_vals:
        raise NotImplementedError("schur with sort_eig_vals is not supported on MPS")
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        return _real_schur(x, compute_schur_vectors=compute_schur_vectors)
    f32 = jnp.float32
    n = x.shape[-1]
    ar = jnp.real(x).astype(f32)
    ai = jnp.imag(x).astype(f32)
    eye = jnp.broadcast_to(jnp.eye(n, dtype=f32), ar.shape)
    if n <= 1:
        outs = [x]
        if compute_schur_vectors:
            outs.append(lax.complex(eye, jnp.zeros_like(eye)).astype(x.dtype))
        return outs

    rvec = jnp.arange(n)
    rows = rvec[:, None]
    cols = rvec[None, :]
    row_axis = ar.ndim - 2
    col_axis = ar.ndim - 1
    tr, ti, tlr, tli, tar, tai = _householder_reduce(ar, ai)
    hess = rows <= cols + 1
    tr = jnp.where(hess, tr, 0.0)
    ti = jnp.where(hess, ti, 0.0)

    def mv(m, v):
        return jnp.einsum("...rc,...c->...r", m, v)

    def acc(i, z):  # Z <- Z H_i, H_i = I - tau v v^H
        zr, zi = z
        vr = jnp.where(rvec == i + 1, 1.0, lax.dynamic_index_in_dim(tlr, i, axis=col_axis, keepdims=False))
        vi = lax.dynamic_index_in_dim(tli, i, axis=col_axis, keepdims=False)
        t_r = lax.dynamic_index_in_dim(tar, i, axis=tar.ndim - 1, keepdims=False)
        t_i = lax.dynamic_index_in_dim(tai, i, axis=tai.ndim - 1, keepdims=False)
        pr, pi = mv(zr, vr) - mv(zi, vi), mv(zr, vi) + mv(zi, vr)
        qr, qi = _cmul(t_r[..., None], t_i[..., None], pr, pi)
        zr = zr - (qr[..., :, None] * vr[..., None, :] + qi[..., :, None] * vi[..., None, :])
        zi = zi - (qi[..., :, None] * vr[..., None, :] - qr[..., :, None] * vi[..., None, :])
        return zr, zi

    zr, zi = lax.fori_loop(0, n - 1, acc, (eye, jnp.zeros_like(eye)))

    batch = ar.shape[:-2]
    eps = float(jnp.finfo(f32).eps)
    norm_t = jnp.sqrt(jnp.sum(ar * ar + ai * ai, axis=(-2, -1)))
    eye2 = jnp.eye(n, dtype=f32)
    maxit = 40 * n

    def ent(m, ea, eb):
        return jnp.sum(m * ea[..., :, None] * eb[..., None, :], axis=(-2, -1))

    def cond(state):
        _, _, _, _, hi, it, _ = state
        return jnp.any(hi > 0) & (it < maxit)

    def body(state):
        tr, ti, zr, zi, hi, it, since = state
        active = hi > 0
        eh = (rvec == hi[..., None]).astype(f32)
        el = (rvec == (hi - 1)[..., None]).astype(f32)
        dr, di = ent(tr, eh, eh), ent(ti, eh, eh)
        a_r, a_i = ent(tr, el, el), ent(ti, el, el)
        br, bi = ent(tr, el, eh), ent(ti, el, eh)
        cr, ci = ent(tr, eh, el), ent(ti, eh, el)
        scale = jnp.hypot(a_r, a_i) + jnp.hypot(dr, di)
        subn = jnp.hypot(cr, ci)
        defl = active & (subn <= 4.0 * eps * (scale + 1e-3 * norm_t))
        do = active & ~defl
        # Wilkinson shift: eigenvalue of the trailing 2x2 block closer to d.
        hr, hi2 = (a_r - dr) / 2, (a_i - di) / 2
        h2r, h2i = _cmul(hr, hi2, hr, hi2)
        bcr, bci = _cmul(br, bi, cr, ci)
        qr, qi = h2r + bcr, h2i + bci
        rq = jnp.hypot(qr, qi)
        sr = jnp.sqrt(jnp.maximum((rq + qr) / 2, 0.0))
        si = jnp.where(qi < 0, -1.0, 1.0) * jnp.sqrt(jnp.maximum((rq - qr) / 2, 0.0))
        mr, mi = (a_r + dr) / 2, (a_i + di) / 2
        d1 = (mr + sr - dr) ** 2 + (mi + si - di) ** 2
        d2 = (mr - sr - dr) ** 2 + (mi - si - di) ** 2
        mur = jnp.where(d1 <= d2, mr + sr, mr - sr)
        mui = jnp.where(d1 <= d2, mi + si, mi - si)
        exceptional = (since % 10) == 9
        mur = jnp.where(exceptional, dr + subn, mur)
        mui = jnp.where(exceptional, di, mui)
        dmask = ((rvec <= hi[..., None]) & do[..., None]).astype(f32)
        shift = eye2 * dmask[..., None, :]
        tr = tr - mur[..., None, None] * shift
        ti = ti - mui[..., None, None] * shift

        cs0 = jnp.zeros((*batch, n - 1), f32)

        def left(k, carry):  # rows k, k+1 <- M (rows); M = [[c, s], [-conj(s), c]]
            tr, ti, cc, csr, csi = carry
            rkr = lax.dynamic_index_in_dim(tr, k, axis=row_axis, keepdims=False)
            rki = lax.dynamic_index_in_dim(ti, k, axis=row_axis, keepdims=False)
            r1r = lax.dynamic_index_in_dim(tr, k + 1, axis=row_axis, keepdims=False)
            r1i = lax.dynamic_index_in_dim(ti, k + 1, axis=row_axis, keepdims=False)
            xr = lax.dynamic_index_in_dim(rkr, k, axis=rkr.ndim - 1, keepdims=False)
            xi = lax.dynamic_index_in_dim(rki, k, axis=rki.ndim - 1, keepdims=False)
            yr = lax.dynamic_index_in_dim(r1r, k, axis=r1r.ndim - 1, keepdims=False)
            yi = lax.dynamic_index_in_dim(r1i, k, axis=r1i.ndim - 1, keepdims=False)
            ax = jnp.hypot(xr, xi)
            nrm = jnp.hypot(ax, jnp.hypot(yr, yi))
            en = do & (k < hi) & (nrm > 0)
            snrm = jnp.where(nrm > 0, nrm, 1.0)
            c = jnp.where(ax > 0, ax / snrm, 0.0)
            uxr = jnp.where(ax > 0, xr / jnp.where(ax > 0, ax, 1.0), 1.0)
            uxi = jnp.where(ax > 0, xi / jnp.where(ax > 0, ax, 1.0), 0.0)
            s_r, s_i = _cmul(uxr, uxi, yr, -yi)
            c = jnp.where(en, c, 1.0)
            s_r = jnp.where(en, s_r / snrm, 0.0)
            s_i = jnp.where(en, s_i / snrm, 0.0)
            ce, sre, sie = c[..., None], s_r[..., None], s_i[..., None]
            t1r, t1i = _cmul(sre, sie, r1r, r1i)
            t2r, t2i = _cmul(-sre, sie, rkr, rki)
            nkr, nki = ce * rkr + t1r, ce * rki + t1i
            n1r, n1i = t2r + ce * r1r, t2i + ce * r1i
            tr = lax.dynamic_update_index_in_dim(tr, nkr[..., None, :], k, axis=row_axis)
            ti = lax.dynamic_update_index_in_dim(ti, nki[..., None, :], k, axis=row_axis)
            tr = lax.dynamic_update_index_in_dim(tr, n1r[..., None, :], k + 1, axis=row_axis)
            ti = lax.dynamic_update_index_in_dim(ti, n1i[..., None, :], k + 1, axis=row_axis)
            cc = lax.dynamic_update_index_in_dim(cc, ce, k, axis=cc.ndim - 1)
            csr = lax.dynamic_update_index_in_dim(csr, sre, k, axis=csr.ndim - 1)
            csi = lax.dynamic_update_index_in_dim(csi, sie, k, axis=csi.ndim - 1)
            return tr, ti, cc, csr, csi

        tr, ti, cc, csr, csi = lax.fori_loop(0, n - 1, left, (tr, ti, cs0 + 1.0, cs0, cs0))

        def rot_cols(mr, mi, k, c, s_r, s_i):  # cols k, k+1 <- (cols) M^H
            ckr = lax.dynamic_index_in_dim(mr, k, axis=col_axis, keepdims=False)
            cki = lax.dynamic_index_in_dim(mi, k, axis=col_axis, keepdims=False)
            c1r = lax.dynamic_index_in_dim(mr, k + 1, axis=col_axis, keepdims=False)
            c1i = lax.dynamic_index_in_dim(mi, k + 1, axis=col_axis, keepdims=False)
            ce, sre, sie = c[..., None], s_r[..., None], s_i[..., None]
            t1r, t1i = _cmul(sre, -sie, c1r, c1i)
            t2r, t2i = _cmul(-sre, -sie, ckr, cki)
            nkr, nki = ce * ckr + t1r, ce * cki + t1i
            n1r, n1i = t2r + ce * c1r, t2i + ce * c1i
            mr = lax.dynamic_update_index_in_dim(mr, nkr[..., :, None], k, axis=col_axis)
            mi = lax.dynamic_update_index_in_dim(mi, nki[..., :, None], k, axis=col_axis)
            mr = lax.dynamic_update_index_in_dim(mr, n1r[..., :, None], k + 1, axis=col_axis)
            mi = lax.dynamic_update_index_in_dim(mi, n1i[..., :, None], k + 1, axis=col_axis)
            return mr, mi

        def right(k, carry):
            tr, ti, zr, zi = carry
            c = lax.dynamic_index_in_dim(cc, k, axis=cc.ndim - 1, keepdims=False)
            s_r = lax.dynamic_index_in_dim(csr, k, axis=csr.ndim - 1, keepdims=False)
            s_i = lax.dynamic_index_in_dim(csi, k, axis=csi.ndim - 1, keepdims=False)
            tr, ti = rot_cols(tr, ti, k, c, s_r, s_i)
            zr, zi = rot_cols(zr, zi, k, c, s_r, s_i)
            return tr, ti, zr, zi

        tr, ti, zr, zi = lax.fori_loop(0, n - 1, right, (tr, ti, zr, zi))
        tr = tr + mur[..., None, None] * shift
        ti = ti + mui[..., None, None] * shift
        dm = (eh[..., :, None] * el[..., None, :] > 0) & defl[..., None, None]
        tr = jnp.where(dm, 0.0, tr)
        ti = jnp.where(dm, 0.0, ti)
        hi = jnp.where(defl, hi - 1, hi)
        since = jnp.where(defl, 0, jnp.where(do, since + 1, since))
        return tr, ti, zr, zi, hi, it + 1, since

    hi0 = jnp.full(batch, n - 1, jnp.int32)
    state = (tr, ti, zr, zi, hi0, jnp.int32(0), jnp.zeros(batch, jnp.int32))
    tr, ti, zr, zi, _, _, _ = _sweep_loop(cond, body, state, maxit, n)
    tr = jnp.where(rows > cols, 0.0, tr)
    ti = jnp.where(rows > cols, 0.0, ti)
    outs = [lax.complex(tr, ti).astype(x.dtype)]
    if compute_schur_vectors:
        outs.append(lax.complex(zr, zi).astype(x.dtype))
    return outs


def _dlanv2(a, b, c, d):
    """LAPACK dlanv2, elementwise: standardise the real 2x2 block [[a, b], [c, d]].

    Returns (a, b, c, d, cs, sn) with old = G new G^T, G = [[cs, -sn], [sn, cs]].
    On return either c = 0 (real eigenvalues) or a = d and b * c < 0.
    """
    import jax.numpy as jnp

    eps = float(jnp.finfo(jnp.float32).eps)

    def sgn(v):
        return jnp.where(v >= 0, 1.0, -1.0)

    def nz(v):
        return jnp.where(v != 0, v, 1.0)

    temp = a - d
    p = 0.5 * temp
    bcmax = jnp.maximum(jnp.abs(b), jnp.abs(c))
    bcmis = jnp.minimum(jnp.abs(b), jnp.abs(c)) * sgn(b) * sgn(c)
    scale = jnp.maximum(jnp.abs(p), bcmax)
    zq = p / nz(scale) * p + bcmax / nz(scale) * bcmis
    # Real eigenvalues: split the block.
    zr = p + sgn(p) * jnp.sqrt(scale) * jnp.sqrt(jnp.maximum(zq, 0.0))
    ra = d + zr
    rd = d - bcmax / nz(zr) * bcmis
    rtau = jnp.hypot(c, zr)
    rcs, rsn = zr / nz(rtau), c / nz(rtau)
    # Complex (or nearly equal real) eigenvalues: equalise the diagonal.
    sigma = b + c
    ctau = jnp.hypot(sigma, temp)
    ccs = jnp.sqrt(0.5 * (1.0 + jnp.abs(sigma) / nz(ctau)))
    csn = -(p / (nz(ctau) * ccs)) * sgn(sigma)
    aa, bb = a * ccs + b * csn, -a * csn + b * ccs
    cc, dd = c * ccs + d * csn, -c * csn + d * ccs
    na, nb = aa * ccs + cc * csn, bb * ccs + dd * csn
    nc, nd = -aa * csn + cc * ccs, -bb * csn + dd * ccs
    mid = 0.5 * (na + nd)
    split = (nc != 0) & (nb != 0) & (sgn(nb) == sgn(nc))
    swap = (nc != 0) & (nb == 0)
    sab, sac = jnp.sqrt(jnp.abs(nb)), jnp.sqrt(jnp.abs(nc))
    pp = sgn(nc) * sab * sac
    tau1 = 1.0 / jnp.sqrt(nz(jnp.abs(nb + nc)))
    cs1, sn1 = sab * tau1, sac * tau1
    ca = jnp.where(split, mid + pp, mid)
    cd = jnp.where(split, mid - pp, mid)
    cb = jnp.where(split, nb - nc, jnp.where(swap, -nc, nb))
    cc2 = jnp.where(split | swap, 0.0, nc)
    ccs2 = jnp.where(split, ccs * cs1 - csn * sn1, jnp.where(swap, -csn, ccs))
    csn2 = jnp.where(split, ccs * sn1 + csn * cs1, jnp.where(swap, ccs, csn))
    real_ev = zq >= 4.0 * eps
    ga = jnp.where(real_ev, ra, ca)
    gb = jnp.where(real_ev, b - c, cb)
    gc = jnp.where(real_ev, 0.0, cc2)
    gd = jnp.where(real_ev, rd, cd)
    gcs = jnp.where(real_ev, rcs, ccs2)
    gsn = jnp.where(real_ev, rsn, csn2)
    # LAPACK case order: c == 0 (done), b == 0 (swap), standard already, general.
    c0 = c == 0
    b0 = ~c0 & (b == 0)
    std = ~c0 & ~b0 & (temp == 0) & (sgn(b) != sgn(c))
    gen = ~(c0 | b0 | std)
    return (
        jnp.where(b0, d, jnp.where(gen, ga, a)),
        jnp.where(b0, -c, jnp.where(gen, gb, b)),
        jnp.where(b0, 0.0, jnp.where(gen, gc, c)),
        jnp.where(b0, a, jnp.where(gen, gd, d)),
        jnp.where(b0, 0.0, jnp.where(gen, gcs, 1.0)),
        jnp.where(b0, 1.0, jnp.where(gen, gsn, 0.0)),
    )


def _real_schur(x, *, compute_schur_vectors):
    """Real Schur decomposition A = Z T Z^T on float32 (LAPACK dhseqr form).

    Hessenberg reduction (accumulating Z), then implicit Francis double-shift
    QR sweeps on the unreduced trailing block [lo, hi]: the first reflector
    comes from the first column of (H - s1 I)(H - s2 I) with the eigenvalues of
    the trailing 2x2 block as shifts, and 3-row reflectors chase the bulge.
    Negligible subdiagonal entries are zeroed; a trailing 1x1 or 2x2 block
    deflates. The dlahqr exceptional shift is used every 10 sweeps without
    deflation; the iteration count is capped at 40 n. Each remaining 2x2 block
    is then standardised with dlanv2: real pairs are split, complex pairs get
    equal diagonal entries and b * c < 0. Batches use masks, no gathers.
    """
    import jax.numpy as jnp
    from jax import lax

    f32 = jnp.float32
    n = x.shape[-1]
    a = x.astype(f32)
    eye = jnp.broadcast_to(jnp.eye(n, dtype=f32), a.shape)
    if n <= 1:
        outs = [x]
        if compute_schur_vectors:
            outs.append(eye.astype(x.dtype))
        return outs

    rvec = jnp.arange(n)
    rows = rvec[:, None]
    cols = rvec[None, :]
    row_axis, col_axis = a.ndim - 2, a.ndim - 1
    t, _, tl, _, taus, _ = _householder_reduce(a, jnp.zeros_like(a))
    t = jnp.where(rows <= cols + 1, t, 0.0)

    def reflect_left(m, v, tau):  # (I - tau v v^T) m
        w = jnp.einsum("...r,...rc->...c", v, m)
        return m - tau[..., None, None] * v[..., :, None] * w[..., None, :]

    def reflect_right(m, v, tau):  # m (I - tau v v^T)
        u = jnp.einsum("...rc,...c->...r", m, v)
        return m - tau[..., None, None] * u[..., :, None] * v[..., None, :]

    def acc(i, z):
        v = jnp.where(rvec == i + 1, 1.0, lax.dynamic_index_in_dim(tl, i, axis=col_axis, keepdims=False))
        tau = lax.dynamic_index_in_dim(taus, i, axis=taus.ndim - 1, keepdims=False)
        return reflect_right(z, v, tau)

    # Without Schur vectors Z is never read: carry a per-batch placeholder and skip every update.
    z = lax.fori_loop(0, n - 1, acc, eye) if compute_schur_vectors else jnp.zeros(a.shape[:-2], f32)

    batch = a.shape[:-2]
    eps = float(jnp.finfo(f32).eps)
    norm_a = jnp.sqrt(jnp.sum(a * a, axis=(-2, -1)))
    maxit = 40 * n
    jvec = jnp.arange(1, n)  # subdiagonal entry (j, j - 1)
    subpos = rows == cols + 1

    def ent(m, i, j):  # m[..., i, j] for per-batch indices (0 when out of range)
        ei = (rvec == i[..., None]).astype(f32)
        ej = (rvec == j[..., None]).astype(f32)
        return jnp.sum(m * ei[..., :, None] * ej[..., None, :], axis=(-2, -1))

    def cond(state):
        _, _, hi, it, _ = state
        return jnp.any(hi > 0) & (it < maxit)

    def body(state):
        t, z, hi, it, since = state
        active = hi > 0
        diag = jnp.diagonal(t, axis1=-2, axis2=-1)
        sub = jnp.diagonal(t, offset=-1, axis1=-2, axis2=-1)
        tst = jnp.abs(diag[..., 1:]) + jnp.abs(diag[..., :-1])
        negl = (jnp.abs(sub) <= 4.0 * eps * (tst + 1e-3 * norm_a[..., None])) & (jvec <= hi[..., None])
        lo = jnp.max(jnp.where(negl, jvec, 0), axis=-1)
        neglc = jnp.concatenate([negl, jnp.zeros_like(negl[..., :1])], axis=-1)  # by column
        t = jnp.where(subpos & neglc[..., None, :], 0.0, t)
        defl1 = active & (lo == hi)
        defl2 = active & (lo == hi - 1)
        do = active & (lo < hi - 1)

        hm1 = hi - 1
        h_hh, h_ll = ent(t, hi, hi), ent(t, hm1, hm1)
        h_lh, h_hl = ent(t, hm1, hi), ent(t, hi, hm1)
        h_ml = ent(t, hm1, hi - 2)
        s = h_ll + h_hh
        p = h_ll * h_hh - h_lh * h_hl
        exceptional = (since % 10) == 9
        ex = jnp.abs(h_hl) + jnp.abs(h_ml)
        e11 = 0.75 * ex + h_hh
        s = jnp.where(exceptional, 2.0 * e11, s)
        p = jnp.where(exceptional, e11 * e11 + 0.4375 * ex * ex, p)

        blk = (rvec >= lo[..., None]) & (rvec <= hi[..., None])
        hb = jnp.where(blk[..., :, None] & blk[..., None, :], t, 0.0)
        elo = (rvec == lo[..., None]).astype(f32)
        c0 = jnp.einsum("...rc,...c->...r", hb, elo)
        m0 = jnp.einsum("...rc,...c->...r", hb, c0) - s[..., None] * c0 + p[..., None] * elo

        def chase(k, carry):
            t, z = carry
            tcol = lax.dynamic_index_in_dim(t, jnp.maximum(k - 1, 0), axis=col_axis, keepdims=False)
            xk = jnp.where((k == lo)[..., None], m0, tcol)
            rmask = (rvec >= k) & (rvec <= k + 2) & (rvec <= hi[..., None])
            xk = jnp.where(rmask, xk, 0.0)
            v, _, tau, _, _, _ = _hh_vector(xk, jnp.zeros_like(xk), k, rvec)
            en = do & (k >= lo) & (k < hi)
            tau = jnp.where(en, tau, 0.0)
            v = jnp.where(en[..., None], v, 0.0)
            t = reflect_right(reflect_left(t, v, tau), v, tau)
            return t, (reflect_right(z, v, tau) if compute_schur_vectors else z)

        t, z = lax.fori_loop(0, n - 1, chase, (t, z))
        t = jnp.where(rows <= cols + 1, t, 0.0)
        hi = jnp.where(defl1, hi - 1, jnp.where(defl2, jnp.maximum(hi - 2, 0), hi))
        since = jnp.where(defl1 | defl2, 0, jnp.where(do, since + 1, since))
        return t, z, hi, it + 1, since

    hi0 = jnp.full(batch, n - 1, jnp.int32)
    state = (t, z, hi0, jnp.int32(0), jnp.zeros(batch, jnp.int32))
    t, z, _, _, _ = _sweep_loop(cond, body, state, maxit, n)
    t = jnp.where(rows > cols + 1, 0.0, t)

    nzs = jnp.diagonal(t, offset=-1, axis1=-2, axis2=-1) != 0
    lead = nzs & ~jnp.concatenate([jnp.zeros_like(nzs[..., :1]), nzs[..., :-1]], axis=-1)

    def at(v, j):
        return lax.dynamic_index_in_dim(v, j, axis=v.ndim - 1, keepdims=False)

    def standardise(j, carry):
        t, z = carry
        en = at(lead, j)
        rj = lax.dynamic_index_in_dim(t, j, axis=row_axis, keepdims=False)
        rj1 = lax.dynamic_index_in_dim(t, j + 1, axis=row_axis, keepdims=False)
        na, nb, nc, nd, cs, sn = _dlanv2(at(rj, j), at(rj, j + 1), at(rj1, j), at(rj1, j + 1))
        cs = jnp.where(en, cs, 1.0)[..., None]
        sn = jnp.where(en, sn, 0.0)[..., None]
        t = lax.dynamic_update_index_in_dim(t, (cs * rj + sn * rj1)[..., None, :], j, axis=row_axis)
        t = lax.dynamic_update_index_in_dim(t, (cs * rj1 - sn * rj)[..., None, :], j + 1, axis=row_axis)

        def rot(m):
            cj = lax.dynamic_index_in_dim(m, j, axis=col_axis, keepdims=False)
            cj1 = lax.dynamic_index_in_dim(m, j + 1, axis=col_axis, keepdims=False)
            m = lax.dynamic_update_index_in_dim(m, (cs * cj + sn * cj1)[..., :, None], j, axis=col_axis)
            return lax.dynamic_update_index_in_dim(m, (cs * cj1 - sn * cj)[..., :, None], j + 1, axis=col_axis)

        t = rot(t)
        if compute_schur_vectors:
            z = rot(z)
        e = en[..., None, None]
        for (r, c), val in (((j, j), na), ((j, j + 1), nb), ((j + 1, j), nc), ((j + 1, j + 1), nd)):
            t = jnp.where(e & (rows == r) & (cols == c), val[..., None, None], t)
        return t, z

    t, z = lax.fori_loop(0, n - 1, standardise, (t, z))
    outs = [t.astype(x.dtype)]
    if compute_schur_vectors:
        outs.append(z.astype(x.dtype))
    return outs


def _normalize_eigvecs(vr, vi):
    """LAPACK geev normalisation: unit 2-norm, largest-magnitude component real."""
    import jax.numpy as jnp

    n = vr.shape[-2]
    nrm = jnp.sqrt(jnp.sum(vr * vr + vi * vi, axis=-2, keepdims=True))
    nrm = jnp.where(nrm > 0, nrm, 1.0)
    vr, vi = vr / nrm, vi / nrm
    idx = jnp.argmax(vr * vr + vi * vi, axis=-2)
    onehot = jnp.arange(n)[:, None] == idx[..., None, :]
    cr = jnp.sum(jnp.where(onehot, vr, 0.0), axis=-2, keepdims=True)
    ci = jnp.sum(jnp.where(onehot, vi, 0.0), axis=-2, keepdims=True)
    cm = jnp.sqrt(cr * cr + ci * ci)
    cm = jnp.where(cm > 0, cm, 1.0)
    pr, pi = cr / cm, -ci / cm
    vr, vi = vr * pr - vi * pi, vr * pi + vi * pr
    return vr, jnp.where(onehot, 0.0, vi)


def _real_to_complex_schur(x):
    """Complex Schur form of a real matrix from its real Schur form (SciPy rsf2csf).

    Each 2x2 block [[a, b], [c, d]] with c != 0 is split by the complex Givens rotation
    G = [[conj(g), s], [-s, g]] whose first row of G^H is the unit eigenvector
    [lambda - d, c] / r for the eigenvalue with positive imaginary part, so T[m-1, m-1]
    becomes that eigenvalue (dhseqr order, as in _real_schur_eigvals). Blocks are
    processed from the bottom; a rotation only mixes entries that are zero outside its
    block rows/columns, so unprocessed blocks stay real. Returns float32 (tr, ti, zr, zi)
    with A = Z T Z^H.
    """
    import jax.numpy as jnp
    from jax import lax

    f32 = jnp.float32
    t, z = _real_schur(x, compute_schur_vectors=True)
    tr, zr = t.astype(f32), z.astype(f32)
    ti, zi = jnp.zeros_like(tr), jnp.zeros_like(zr)
    n = tr.shape[-1]
    row_axis, col_axis = tr.ndim - 2, tr.ndim - 1
    rvec = jnp.arange(n)
    rows, cols = rvec[:, None], rvec[None, :]

    def at(m, i, j):
        r = lax.dynamic_index_in_dim(m, i, axis=row_axis, keepdims=False)
        return lax.dynamic_index_in_dim(r, j, axis=r.ndim - 1, keepdims=False)

    def cmul(ar, ai, br, bi):
        return ar * br - ai * bi, ar * bi + ai * br

    def rot_rows(mr, mi, i, gr, gi, s):  # rows (i, i+1) <- G @ rows
        r1r = lax.dynamic_index_in_dim(mr, i, axis=row_axis, keepdims=False)
        r1i = lax.dynamic_index_in_dim(mi, i, axis=row_axis, keepdims=False)
        r2r = lax.dynamic_index_in_dim(mr, i + 1, axis=row_axis, keepdims=False)
        r2i = lax.dynamic_index_in_dim(mi, i + 1, axis=row_axis, keepdims=False)
        ar, ai = cmul(gr, -gi, r1r, r1i)
        br, bi = cmul(gr, gi, r2r, r2i)
        n1r, n1i = ar + s * r2r, ai + s * r2i
        n2r, n2i = br - s * r1r, bi - s * r1i
        mr = lax.dynamic_update_index_in_dim(mr, n1r[..., None, :], i, axis=row_axis)
        mi = lax.dynamic_update_index_in_dim(mi, n1i[..., None, :], i, axis=row_axis)
        mr = lax.dynamic_update_index_in_dim(mr, n2r[..., None, :], i + 1, axis=row_axis)
        mi = lax.dynamic_update_index_in_dim(mi, n2i[..., None, :], i + 1, axis=row_axis)
        return mr, mi

    def rot_cols(mr, mi, i, gr, gi, s):  # cols (i, i+1) <- cols @ G^H
        c1r = lax.dynamic_index_in_dim(mr, i, axis=col_axis, keepdims=False)
        c1i = lax.dynamic_index_in_dim(mi, i, axis=col_axis, keepdims=False)
        c2r = lax.dynamic_index_in_dim(mr, i + 1, axis=col_axis, keepdims=False)
        c2i = lax.dynamic_index_in_dim(mi, i + 1, axis=col_axis, keepdims=False)
        ar, ai = cmul(gr, gi, c1r, c1i)
        br, bi = cmul(gr, -gi, c2r, c2i)
        n1r, n1i = ar + s * c2r, ai + s * c2i
        n2r, n2i = br - s * c1r, bi - s * c1i
        mr = lax.dynamic_update_index_in_dim(mr, n1r[..., :, None], i, axis=col_axis)
        mi = lax.dynamic_update_index_in_dim(mi, n1i[..., :, None], i, axis=col_axis)
        mr = lax.dynamic_update_index_in_dim(mr, n2r[..., :, None], i + 1, axis=col_axis)
        mi = lax.dynamic_update_index_in_dim(mi, n2i[..., :, None], i + 1, axis=col_axis)
        return mr, mi

    def body(k, carry):
        tr, ti, zr, zi = carry
        m = n - 1 - k
        a, b = at(tr, m - 1, m - 1), at(tr, m - 1, m)
        c, d = at(tr, m, m - 1), at(tr, m, m)
        en = c != 0
        p = 0.5 * (a - d)
        disc = p * p + b * c
        root = jnp.sqrt(jnp.abs(disc))
        mur = jnp.where(disc < 0, p, p + jnp.where(p >= 0, root, -root))  # lambda - d
        mui = jnp.where(disc < 0, root, 0.0)
        r = jnp.sqrt(mur * mur + mui * mui + c * c)
        r = jnp.where(en & (r > 0), r, 1.0)
        gr = jnp.where(en, mur / r, 1.0)[..., None]
        gi = jnp.where(en, mui / r, 0.0)[..., None]
        s = jnp.where(en, c / r, 0.0)[..., None]
        tr, ti = rot_rows(tr, ti, m - 1, gr, gi, s)
        tr, ti = rot_cols(tr, ti, m - 1, gr, gi, s)
        zr, zi = rot_cols(zr, zi, m - 1, gr, gi, s)
        sub = (rows == m) & (cols == m - 1)
        return jnp.where(sub, 0.0, tr), jnp.where(sub, 0.0, ti), zr, zi

    if n > 1:
        tr, ti, zr, zi = lax.fori_loop(0, n - 1, body, (tr, ti, zr, zi))
    return tr, ti, zr, zi


def _real_schur_eigvals(x):
    """Eigenvalues of a real matrix from its standardised real Schur form (dhseqr order).

    A 1x1 block gives t_jj. After dlanv2 a 2x2 block [[a, b], [c, a]] has b * c < 0 and
    gives a +/- i sqrt(|b|) sqrt(|c|), positive imaginary part first. Schur vectors are
    not formed. Returns (real, imag) float32 parts.
    """
    import jax.numpy as jnp

    (t,) = _real_schur(x, compute_schur_vectors=False)
    t = t.astype(jnp.float32)
    d = jnp.diagonal(t, axis1=-2, axis2=-1)
    sub = jnp.diagonal(t, offset=-1, axis1=-2, axis2=-1)  # t[j + 1, j]
    sup = jnp.diagonal(t, offset=1, axis1=-2, axis2=-1)  # t[j, j + 1]
    im = jnp.where(sub != 0, jnp.sqrt(jnp.abs(sup)) * jnp.sqrt(jnp.abs(sub)), 0.0)
    pad = jnp.zeros_like(d[..., :1])
    wi = jnp.concatenate([im, pad], axis=-1) - jnp.concatenate([pad, im], axis=-1)
    return d, wi


def _eig(x, *, compute_left_eigenvectors, compute_right_eigenvectors, **_):
    """General eigendecomposition from the complex Schur form A = Z T Z^H.

    Eigenvalues are diag(T). Right eigenvectors: backward substitution
    (T - t_kk I) x = 0 with x_k = 1, all k at once, then V = Z X. Left
    eigenvectors: forward substitution on T^H, then U = Z Y. Pivots smaller
    than eps ||T|| are floored as in LAPACK trevc. Vectors use the geev
    normalisation. Order of eigenvalues follows the Schur form.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax._src import dtypes

    out_dtype = dtypes.to_complex_dtype(x.dtype)
    f32 = jnp.float32
    n = x.shape[-1]
    if not (compute_left_eigenvectors or compute_right_eigenvectors) and not dtypes.issubdtype(
        x.dtype, jnp.complexfloating
    ):
        # Real input, eigenvalues only: the real Schur form is ~3.5x cheaper than the complex one.
        wr, wi = _real_schur_eigvals(x)
        return [lax.complex(wr, wi).astype(out_dtype)]
    if dtypes.issubdtype(x.dtype, jnp.complexfloating):
        t, z = _complex_schur(x.astype(jnp.complex64), compute_schur_vectors=True)
    else:
        # Real input with eigenvectors: real Schur + rsf2csf instead of the complex QR iteration.
        s_tr, s_ti, s_zr, s_zi = _real_to_complex_schur(x)
        t, z = lax.complex(s_tr, s_ti), lax.complex(s_zr, s_zi)
    w = jnp.diagonal(t, axis1=-2, axis2=-1)
    outs = [w.astype(out_dtype)]
    if not (compute_left_eigenvectors or compute_right_eigenvectors):
        return outs
    tr, ti = jnp.real(t).astype(f32), jnp.imag(t).astype(f32)
    zr, zi = jnp.real(z).astype(f32), jnp.imag(z).astype(f32)
    lr, li = jnp.real(w).astype(f32), jnp.imag(w).astype(f32)
    rvec = jnp.arange(n)
    row_axis, col_axis = tr.ndim - 2, tr.ndim - 1
    norm_t = jnp.sqrt(jnp.sum(tr * tr + ti * ti, axis=(-2, -1)))
    smin = jnp.maximum(float(jnp.finfo(f32).eps) * norm_t, 1e-30)[..., None]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=f32), tr.shape)

    def solve_entry(sr, si, dr, di):  # -(s / d) with |d| floored
        small = jnp.sqrt(dr * dr + di * di) < smin
        dr = jnp.where(small, smin, dr)
        di = jnp.where(small, 0.0, di)
        den = dr * dr + di * di
        return -(sr * dr + si * di) / den, -(si * dr - sr * di) / den

    def mm(ar, ai, br, bi):
        return ar @ br - ai @ bi, ar @ bi + ai @ br

    def finish(yr, yi):
        vr, vi = mm(zr, zi, yr, yi)
        vr, vi = _normalize_eigvecs(vr, vi)
        return lax.complex(vr, vi).astype(out_dtype)

    vl = vrt = None
    if compute_right_eigenvectors:

        def rbody(s, xs):
            xr, xi = xs
            i = n - 2 - s
            rr = jnp.where(rvec > i, lax.dynamic_index_in_dim(tr, i, axis=row_axis, keepdims=False), 0.0)
            ri = jnp.where(rvec > i, lax.dynamic_index_in_dim(ti, i, axis=row_axis, keepdims=False), 0.0)
            sr = jnp.einsum("...j,...jk->...k", rr, xr) - jnp.einsum("...j,...jk->...k", ri, xi)
            si = jnp.einsum("...j,...jk->...k", rr, xi) + jnp.einsum("...j,...jk->...k", ri, xr)
            tir = lax.dynamic_index_in_dim(lr, i, axis=lr.ndim - 1, keepdims=True)
            tii = lax.dynamic_index_in_dim(li, i, axis=li.ndim - 1, keepdims=True)
            yr, yi = solve_entry(sr, si, tir - lr, tii - li)
            cur_r = lax.dynamic_index_in_dim(xr, i, axis=row_axis, keepdims=False)
            cur_i = lax.dynamic_index_in_dim(xi, i, axis=row_axis, keepdims=False)
            nr = jnp.where(rvec > i, yr, cur_r)
            ni = jnp.where(rvec > i, yi, cur_i)
            xr = lax.dynamic_update_index_in_dim(xr, nr[..., None, :], i, axis=row_axis)
            xi = lax.dynamic_update_index_in_dim(xi, ni[..., None, :], i, axis=row_axis)
            return xr, xi

        xr, xi = lax.fori_loop(0, n - 1, rbody, (eye, jnp.zeros_like(eye)))
        vrt = finish(xr, xi)
    if compute_left_eigenvectors:

        def lbody(i, ys):
            yr, yi = ys
            cr = jnp.where(rvec < i, lax.dynamic_index_in_dim(tr, i, axis=col_axis, keepdims=False), 0.0)
            ci = jnp.where(rvec < i, -lax.dynamic_index_in_dim(ti, i, axis=col_axis, keepdims=False), 0.0)
            sr = jnp.einsum("...j,...jk->...k", cr, yr) - jnp.einsum("...j,...jk->...k", ci, yi)
            si = jnp.einsum("...j,...jk->...k", cr, yi) + jnp.einsum("...j,...jk->...k", ci, yr)
            tir = lax.dynamic_index_in_dim(lr, i, axis=lr.ndim - 1, keepdims=True)
            tii = lax.dynamic_index_in_dim(li, i, axis=li.ndim - 1, keepdims=True)
            nr_, ni_ = solve_entry(sr, si, tir - lr, -(tii - li))
            cur_r = lax.dynamic_index_in_dim(yr, i, axis=row_axis, keepdims=False)
            cur_i = lax.dynamic_index_in_dim(yi, i, axis=row_axis, keepdims=False)
            nr = jnp.where(rvec < i, nr_, cur_r)
            ni = jnp.where(rvec < i, ni_, cur_i)
            yr = lax.dynamic_update_index_in_dim(yr, nr[..., None, :], i, axis=row_axis)
            yi = lax.dynamic_update_index_in_dim(yi, ni[..., None, :], i, axis=row_axis)
            return yr, yi

        yr, yi = lax.fori_loop(1, n, lbody, (eye, jnp.zeros_like(eye)))
        vl = finish(yr, yi)
    if compute_left_eigenvectors:
        outs.append(vl)
    if compute_right_eigenvectors:
        outs.append(vrt)
    return outs


def register() -> None:
    """Register the MPS lowerings. Call after the 'mps' plugin is registered."""
    from jax._src.interpreters import mlir
    from jax._src.lax import linalg as lax_linalg

    mlir.register_lowering(lax_linalg.eigh_p, mlir.lower_fun(_jacobi_eigh, multiple_results=True), platform="mps")
    # JAX's generic svd lowering runs QDWH, which calls stablehlo.cholesky
    # inside while bodies; cholesky is a native-only op on MPS and cannot run
    # there, so svd/pinv/lstsq failed. One-sided Jacobi avoids it.
    mlir.register_lowering(lax_linalg.svd_p, mlir.lower_fun(_jacobi_svd, multiple_results=True), platform="mps")

    # The native triangular_solve / cholesky kernels are float-only. Complex
    # inputs lower to the pure-JAX versions above; real inputs keep JAX's
    # default rule (the native op) unchanged.
    complex_tri = mlir.lower_fun(_complex_triangular_solve, multiple_results=False)
    complex_chol = mlir.lower_fun(_complex_cholesky, multiple_results=False)

    def _triangular_solve_rule(ctx, a, b, **params):
        if np.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
            return complex_tri(ctx, a, b, **params)
        return lax_linalg._triangular_solve_lowering(ctx, a, b, **params)

    def _cholesky_rule(ctx, x):
        if np.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
            return complex_chol(ctx, x)
        return lax_linalg._cholesky_lowering(ctx, x)

    mlir.register_lowering(lax_linalg.triangular_solve_p, _triangular_solve_rule, platform="mps")
    mlir.register_lowering(lax_linalg.cholesky_p, _cholesky_rule, platform="mps")

    # The native Qr / ProductOfElementaryHouseholderReflectors handlers are
    # float32-only; complex QR lowers to the LAPACK-convention Householder above.
    complex_geqrf = mlir.lower_fun(_complex_geqrf, multiple_results=True)
    complex_hprod = mlir.lower_fun(_complex_householder_product, multiple_results=False)

    def _geqrf_rule(ctx, operand):
        if np.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
            return complex_geqrf(ctx, operand)
        return lax_linalg._geqrf_lowering_rule(ctx, operand)

    def _householder_product_rule(ctx, a, taus):
        if np.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
            return complex_hprod(ctx, a, taus)
        return lax_linalg._householder_product_lowering(ctx, a, taus)

    mlir.register_lowering(lax_linalg.geqrf_p, _geqrf_rule, platform="mps")
    mlir.register_lowering(lax_linalg.householder_product_p, _householder_product_rule, platform="mps")

    # hessenberg / tridiagonal had no MPS lowering at all (CPU/GPU LAPACK only).
    mlir.register_lowering(lax_linalg.hessenberg_p, mlir.lower_fun(_hessenberg, multiple_results=True), platform="mps")
    mlir.register_lowering(lax_linalg.tridiagonal_p, mlir.lower_fun(_tridiagonal, multiple_results=True), platform="mps")
    # schur had no MPS lowering; complex Schur (enough for sqrtm) via shifted QR.
    mlir.register_lowering(lax_linalg.schur_p, mlir.lower_fun(_complex_schur, multiple_results=True), platform="mps")
    # eig had no MPS lowering; eigenvalues/vectors from the complex Schur form.
    mlir.register_lowering(lax_linalg.eig_p, mlir.lower_fun(_eig, multiple_results=True), platform="mps")
