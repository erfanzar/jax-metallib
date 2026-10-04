# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scatter with several scatter batch dims must match XLA:CPU.

The gradient of an embedding lookup with a 2-D index array is a scatter-add with
indices [B, T, 1] and updates [B, T, F]. The lowering handed MPSGraph updates of
higher rank than the data and the process aborted ("updates rank and result rank
must be equal"), so every case runs in its own subprocess.
"""

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _case(name):
    rng = np.random.default_rng(3)
    if name == "embed_grad_2d_idx":
        t = rng.standard_normal((100, 5)).astype(np.float32)
        i = rng.integers(0, 100, (3, 4)).astype(np.int32)
        w = jnp.arange(5, dtype=jnp.float32)
        return jax.grad(lambda t, i: jnp.sum(t[i] * w)), (t, i)
    if name == "embed_grad_3d_idx":
        t = rng.standard_normal((50, 7)).astype(np.float32)
        i = rng.integers(0, 50, (2, 3, 4)).astype(np.int32)
        return jax.grad(lambda t, i: jnp.sum(jnp.sin(t[i]))), (t, i)
    if name == "add_2d_idx_duplicates":
        t = rng.standard_normal((20, 6)).astype(np.float32)
        i = rng.integers(0, 20, (4, 5)).astype(np.int32)
        u = rng.standard_normal((4, 5, 6)).astype(np.float32)
        return (lambda t, i, u: t.at[i].add(u)), (t, i, u)
    if name == "add_1d_table_2d_idx":
        t = rng.standard_normal((30,)).astype(np.float32)
        i = rng.integers(0, 30, (4, 5)).astype(np.int32)
        u = rng.standard_normal((4, 5)).astype(np.float32)
        return (lambda t, i, u: t.at[i].add(u)), (t, i, u)
    if name == "set_unique_2d_idx":
        t = rng.standard_normal((40, 3)).astype(np.float32)
        i = rng.permutation(40)[:12].reshape(3, 4).astype(np.int32)
        u = rng.standard_normal((3, 4, 3)).astype(np.float32)
        return (lambda t, i, u: t.at[i].set(u)), (t, i, u)
    if name == "max_unique_2d_idx":
        t = rng.standard_normal((40, 3)).astype(np.float32)
        i = rng.permutation(40)[:12].reshape(3, 4).astype(np.int32)
        u = rng.standard_normal((3, 4, 3)).astype(np.float32)
        return (lambda t, i, u: t.at[i].max(u)), (t, i, u)
    if name == "add_2d_idx_in_loop":
        t = rng.standard_normal((20, 6)).astype(np.float32)
        i = rng.integers(0, 20, (4, 5)).astype(np.int32)
        u = rng.standard_normal((4, 5, 6)).astype(np.float32)
        return (lambda t, i, u: lax.fori_loop(0, 3, lambda k, acc: acc.at[i].add(u), t)), (t, i, u)
    if name == "grad_axis1_2d_idx":
        t = rng.standard_normal((6, 40)).astype(np.float32)
        i = rng.integers(0, 40, (3, 4)).astype(np.int32)
        return jax.grad(lambda t, i: jnp.sum(jnp.sin(t[:, i]))), (t, i)
    if name == "grad_axis2_2d_idx":
        t = rng.standard_normal((2, 5, 30)).astype(np.float32)
        i = rng.integers(0, 30, (3, 4)).astype(np.int32)
        return jax.grad(lambda t, i: jnp.sum(jnp.cos(t[..., i]))), (t, i)
    if name == "grad_take_axis1_2d_idx":
        t = rng.standard_normal((6, 40)).astype(np.float32)
        i = rng.integers(0, 40, (3, 4)).astype(np.int32)
        return jax.grad(lambda t, i: jnp.sum(jnp.take(t, i, axis=1) ** 2)), (t, i)
    if name == "add_axis1_2d_idx_duplicates":
        t = rng.standard_normal((6, 40)).astype(np.float32)
        i = rng.integers(0, 40, (3, 4)).astype(np.int32)
        u = rng.standard_normal((6, 3, 4)).astype(np.float32)
        return (lambda t, i, u: t.at[:, i].add(u)), (t, i, u)
    if name == "set_axis1_unique_2d_idx":
        t = rng.standard_normal((6, 40)).astype(np.float32)
        i = rng.permutation(40)[:12].reshape(3, 4).astype(np.int32)
        u = rng.standard_normal((6, 3, 4)).astype(np.float32)
        return (lambda t, i, u: t.at[:, i].set(u)), (t, i, u)
    if name == "grad_vmap_row_gather_2d_idx":
        t = rng.standard_normal((4, 30, 5)).astype(np.float32)
        i = rng.integers(0, 30, (4, 3, 2)).astype(np.int32)
        return jax.grad(lambda t, i: jnp.sum(jnp.sin(jax.vmap(lambda a, b: a[b])(t, i)))), (t, i)
    if name == "vmap_add_2d_idx_duplicates":
        t = rng.standard_normal((4, 30, 5)).astype(np.float32)
        i = rng.integers(0, 30, (4, 3, 2)).astype(np.int32)
        u = rng.standard_normal((4, 3, 2, 5)).astype(np.float32)
        return (lambda t, i, u: jax.vmap(lambda a, b, c: a.at[b].add(c))(t, i, u)), (t, i, u)
    if name == "add_axis1_2d_idx_f16":
        # Unique indices: with duplicates the f16 accumulation order is
        # implementation-defined and legitimately differs by a few ulps.
        t = rng.standard_normal((6, 40)).astype(np.float16)
        i = rng.permutation(40)[:12].reshape(3, 4).astype(np.int32)
        u = rng.standard_normal((6, 3, 4)).astype(np.float16)
        return (lambda t, i, u: t.at[:, i].add(u)), (t, i, u)
    if name == "vmap_add_2d_idx_i32":
        t = rng.integers(-50, 50, (4, 30, 5)).astype(np.int32)
        i = rng.integers(0, 30, (4, 3, 2)).astype(np.int32)
        u = rng.integers(-50, 50, (4, 3, 2, 5)).astype(np.int32)
        return (lambda t, i, u: jax.vmap(lambda a, b, c: a.at[b].add(c))(t, i, u)), (t, i, u)
    raise KeyError(name)


_CASES = [
    "embed_grad_2d_idx",
    "embed_grad_3d_idx",
    "add_2d_idx_duplicates",
    "add_1d_table_2d_idx",
    "set_unique_2d_idx",
    "max_unique_2d_idx",
    "add_2d_idx_in_loop",
    "grad_axis1_2d_idx",
    "grad_axis2_2d_idx",
    "grad_take_axis1_2d_idx",
    "add_axis1_2d_idx_duplicates",
    "set_axis1_unique_2d_idx",
    "grad_vmap_row_gather_2d_idx",
    "vmap_add_2d_idx_duplicates",
    "add_axis1_2d_idx_f16",
    "vmap_add_2d_idx_i32",
]

_CHILD = textwrap.dedent(
    """
    import sys
    import jax
    import numpy as np
    from jax import numpy as jnp
    sys.path.insert(0, sys.argv[2])
    import test_scatter_multi_batch as t
    fn, args = t._case(sys.argv[1])
    expected = np.asarray(jax.jit(fn, backend="cpu")(*args))
    f = jax.jit(fn) if sys.argv[3] == "jit" else fn
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(f(*[jnp.asarray(a) for a in args]))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("OK")
    """
)


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize("case", _CASES)
def test_scatter_multi_batch_matches_cpu(case, mode):
    env = {**os.environ, "JAX_PLATFORMS": "cpu,mps"}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, case, os.path.dirname(__file__), mode],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (proc.returncode, proc.stderr[-1500:])
