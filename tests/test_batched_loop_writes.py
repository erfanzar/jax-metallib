# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Counted loops whose only buffer writes are full-row dynamic_update_slices.

The plugin unrolls such loops and applies U row writes per MPSGraph iteration
in one pass instead of U full-buffer writes. These cases pin down ordering
(later writes win), XLA start clamping, stacked scan outputs (forward and
reverse, one or two outputs), and the shapes that must fall back.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_RNG = np.random.default_rng(9)


def _row_loop(index_fn, steps, rows=10, cols=6):
    def f(buf, src):
        def body(i, b):
            return lax.dynamic_update_slice(b, src[i][None] * (i + 1).astype(b.dtype), (index_fn(i), 0))

        return lax.fori_loop(0, steps, body, buf)

    buf = _RNG.standard_normal((rows, cols)).astype(np.float32)
    src = _RNG.standard_normal((steps, cols)).astype(np.float32)
    return f, (buf, src)


def _axis_loop(axis, index_fn, steps, shape):
    # One slice along `axis` per step (e.g. a [1, T, H, D] KV-cache fill).
    def f(buf, src):
        def body(i, b):
            upd = jnp.expand_dims(src[i], axis) * (i + 1).astype(b.dtype)
            starts = [jnp.int32(0)] * b.ndim
            starts[axis] = index_fn(i)
            return lax.dynamic_update_slice(b, upd, tuple(starts))

        return lax.fori_loop(0, steps, body, buf)

    buf = _RNG.standard_normal(shape).astype(np.float32)
    src_shape = (steps, *(s for d, s in enumerate(shape) if d != axis))
    src = _RNG.standard_normal(src_shape).astype(np.float32)
    return f, (buf, src)


def _rnn(stacked_outputs, reverse=False, steps=64, dtype=np.float32):
    def f(w, xs):
        def step(h, x):
            h = jnp.tanh(h @ w + x)
            return h, ((h, h * 2.0) if stacked_outputs == 2 else h)

        return lax.scan(step, jnp.zeros((4, 16), xs.dtype), xs, reverse=reverse)[1]

    w = (_RNG.standard_normal((16, 16)) * 0.2).astype(dtype)
    xs = _RNG.standard_normal((steps, 4, 16)).astype(dtype)
    return f, (w, xs)


_CASES = {
    "scan_stacked": lambda: _rnn(1),
    "scan_two_outputs": lambda: _rnn(2),
    "scan_reverse": lambda: _rnn(1, reverse=True),
    "scan_steps_not_divisible": lambda: _rnn(1, steps=13),
    "scan_f16_fallback": lambda: _rnn(1, dtype=np.float16),
    "rows_in_order": lambda: _row_loop(lambda i: i, 8, rows=8),
    "rows_duplicate_last_wins": lambda: _row_loop(lambda i: i // 3, 16),
    "rows_clamped_high": lambda: _row_loop(lambda i: i + 5, 16),
    "rows_clamped_negative": lambda: _row_loop(lambda i: i - 6, 16),
    "kv_cache_axis1": lambda: _axis_loop(1, lambda i: i, 64, (1, 64, 4, 8)),
    "axis1_duplicate_last_wins": lambda: _axis_loop(1, lambda i: i // 3, 16, (6, 10)),
    "axis1_clamped_high": lambda: _axis_loop(1, lambda i: i + 5, 16, (6, 10)),
    "axis2_3d": lambda: _axis_loop(2, lambda i: i, 12, (2, 3, 12)),
    "scalar_rows": lambda: (
        lambda xs: lax.scan(lambda c, x: (c + x, c * x), jnp.float32(1.0), xs)[1],
        (_RNG.standard_normal((32,)).astype(np.float32),),
    ),
}


def _reads_buffer():
    # The body reads the buffer being written, so it must not be batched.
    def f(buf):
        def body(i, b):
            return lax.dynamic_update_slice(b, b[(i + 1) % 8][None] + 1.0, (i, 0))

        return lax.fori_loop(0, 16, body, buf)

    return f, (_RNG.standard_normal((8, 5)).astype(np.float32),)


_CASES["body_reads_buffer"] = _reads_buffer


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("case", list(_CASES))
def test_batched_loop_writes_match_cpu(case, jit):
    fn, args = _CASES[case]()
    expected = jax.tree.map(np.asarray, jax.jit(fn, backend="cpu")(*args))
    f = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("mps")[0]):
        actual = jax.tree.map(np.asarray, f(*[jnp.asarray(a) for a in args]))
    tol = 2e-3 if case == "scan_f16_fallback" else 1e-5
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.shape == e.shape and a.dtype == e.dtype
        np.testing.assert_allclose(a.astype(np.float32), e.astype(np.float32), rtol=tol, atol=tol)
