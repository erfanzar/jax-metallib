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


def _jacobi_eigh(x, *, lower, sort_eigenvalues, subset_by_index, algorithm):
    import jax.numpy as jnp
    from jax import lax

    del algorithm
    n = x.shape[-1]
    if subset_by_index is not None and tuple(subset_by_index) != (0, n):
        raise NotImplementedError("eigh with subset_by_index is not supported on MPS")
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise NotImplementedError("complex eigh is not supported on MPS")

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


def _jacobi_svd(x, *, full_matrices, compute_uv, subset_by_index, algorithm=None):
    import jax.numpy as jnp

    del algorithm
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise NotImplementedError("complex svd is not supported on MPS")
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


def register() -> None:
    """Register the MPS lowerings. Call after the 'mps' plugin is registered."""
    from jax._src.interpreters import mlir
    from jax._src.lax import linalg as lax_linalg

    mlir.register_lowering(lax_linalg.eigh_p, mlir.lower_fun(_jacobi_eigh, multiple_results=True), platform="mps")
    # JAX's generic svd lowering runs QDWH, which calls stablehlo.cholesky
    # inside while bodies; cholesky is a native-only op on MPS and cannot run
    # there, so svd/pinv/lstsq failed. One-sided Jacobi avoids it.
    mlir.register_lowering(lax_linalg.svd_p, mlir.lower_fun(_jacobi_svd, multiple_results=True), platform="mps")
