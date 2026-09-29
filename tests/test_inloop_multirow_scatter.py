# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Multi-row window scatter-set inside loop bodies must not abort on MPS.

The in-control-flow scatter-set lowering handled only a single index: K row
updates (a.at[idx].set(rows) in a fori_loop, and the row swaps inside
lax.linalg.lu) aborted the process on a broadcast-incompatible select, and a
single index whose updates kept a size-1 batch dim aborted on an incompatible
reshape (lu of a 2x2 matrix). Each case runs in a child process so an abort is
reported as a test failure instead of killing pytest.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    name, jit = sys.argv[1], sys.argv[2] == "jit"
    rng = np.random.default_rng(7)
    m3 = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]], np.float32)

    def row_swap_loop(a):
        def body(i, a):
            idx = jnp.stack([i, i + 1])
            return a.at[idx].set(a[idx[::-1]])

        return lax.fori_loop(0, a.shape[0] - 1, body, a)

    def row_tail_loop(a):
        def body(i, a):
            idx = jnp.minimum(jnp.arange(2, dtype=jnp.int32) + i, a.shape[0] - 1)
            return a.at[idx].set(a[idx] * 2.0, unique_indices=False)

        return lax.fori_loop(0, 2, body, a)

    def three_rows_loop(a):
        def body(i, a):
            idx = (jnp.arange(3, dtype=jnp.int32) * 2 + i) % a.shape[0]
            return a.at[idx].set(a[idx] + i.astype(a.dtype))

        return lax.fori_loop(0, 3, body, a)

    def int_row_swap_loop(a):
        return row_swap_loop(a)

    cases = {
        "lu_2": (lambda a: lax.linalg.lu(a), m3[:2, :2]),
        "lu_3": (lambda a: lax.linalg.lu(a), m3),
        "lu_6": (lambda a: lax.linalg.lu(a), rng.standard_normal((6, 6)).astype(np.float32)),
        "lu_solve_5": (
            lambda a: jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(a), jnp.ones((a.shape[0],), a.dtype)),
            (rng.standard_normal((5, 5)) + 5 * np.eye(5)).astype(np.float32),
        ),
        "row_swap": (row_swap_loop, rng.standard_normal((5, 4)).astype(np.float32)),
        "row_tail": (row_tail_loop, m3),
        "three_rows": (three_rows_loop, rng.standard_normal((7, 3)).astype(np.float32)),
        "int_row_swap": (int_row_swap_loop, (np.arange(20, dtype=np.int32) * 16777219).reshape(5, 4)),
    }
    fn, x = cases[name]
    f = jax.jit(fn) if jit else fn
    out = {}
    for p in ("cpu", "mps"):
        d = jax.devices(p)[0]
        with jax.default_device(d):
            out[p] = [np.asarray(r) for r in jax.tree_util.tree_leaves(f(jax.device_put(x, d)))]
    for c, m in zip(out["cpu"], out["mps"], strict=True):
        assert c.dtype == m.dtype and c.shape == m.shape, (c.dtype, m.dtype, c.shape, m.shape)
        if np.issubdtype(c.dtype, np.floating):
            assert np.allclose(c, m, rtol=1e-4, atol=1e-5), (c, m)
        else:
            assert np.array_equal(c, m), (c, m)
    print("OK")
    """
)

_CASES = ["lu_2", "lu_3", "lu_6", "lu_solve_5", "row_swap", "row_tail", "three_rows", "int_row_swap"]


@pytest.mark.parametrize("jit", ["jit", "eager"])
@pytest.mark.parametrize("case", _CASES)
def test_inloop_multirow_scatter_matches_cpu(case, jit):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, case, jit],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert r.returncode == 0 and "OK" in r.stdout, (r.returncode, r.stderr[-800:])
