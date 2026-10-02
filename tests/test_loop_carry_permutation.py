# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Loop bodies that forward one carry into another slot, e.g. (a, b) -> (b, a + b).

MPSGraph loop bodies whose output is another slot's iteration argument read
that argument after it had already been updated, so a swap or shift of carries
silently computed with the new value (dynamic-bound Fibonacci, carry shifts
next to dynamic_update_slice, 3-cycles under unroll 4). Compared exactly to CPU.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _on(platform, fn, *xs):
    dev = jax.devices(platform)[0]
    f = jax.jit(fn, device=dev)
    return jax.tree_util.tree_map(np.asarray, f(*[jax.device_put(x, dev) for x in xs]))


def fib_dynamic(n):  # dynamic trip count: never unrolled
    return lax.fori_loop(0, n, lambda _, c: (c[1], c[0] + c[1]), (jnp.float32(0.0), jnp.float32(1.0)))


def fib_dynamic_int(n):
    return lax.fori_loop(0, n, lambda _, c: (c[1], c[0] + c[1]), (jnp.int32(0), jnp.int32(1)))


def shift_with_row_write(x):  # constant trip count, but the row write disables unrolling
    def body(k, carry):
        x, a, b = carry
        r = lax.dynamic_index_in_dim(x, k, axis=0, keepdims=False)
        x = lax.dynamic_update_index_in_dim(x, (r * b + a)[None, :], k, axis=0)
        return x, k.astype(jnp.float32) + 2.0, a

    x, a, b = lax.fori_loop(0, 8, body, (x, jnp.float32(0.5), jnp.float32(1.5)))
    return x, a, b


def three_cycle(x):  # 8 trips -> unroll 4; 4 steps of a 3-cycle is not the identity
    def body(_, c):
        a, b, d = c
        return b, d, a * 2.0 + 1.0

    return lax.fori_loop(0, 8, body, (x, x + 1.0, x - 1.0))


def while_swap(x):  # value-dependent condition: MPSGraph while, not the counted for-loop
    def cond(c):
        return c[0] < 1000.0

    def body(c):
        a, b, k = c
        return b + a, a, k + 1

    return lax.while_loop(cond, body, (x, x * 0.0 + 1.0, jnp.int32(0)))


def _dyn8(x):  # data-dependent trip count 8: the loop is never unrolled
    return (jnp.ravel(x)[0] * 0.0 + 8.0).astype(jnp.int32)


def forward_up_dynamic(x):  # slot 0's old value forwarded into slot 1
    def body(_, c):
        a, _b = c
        return a * 2.0 + 1.0, a

    return lax.fori_loop(0, _dyn8(x), body, (x, x + 3.0))


def forward_up_through_add_zero(x):  # a + 0.0 is folded back to a by MPSGraph
    def body(_, c):
        a, _b = c
        return a * 2.0 + 1.0, a + 0.0

    return lax.fori_loop(0, _dyn8(x), body, (x, x + 3.0))


def forward_down_dynamic(x):  # slot 1 forwarded into slot 0 (was already right)
    def body(_, c):
        _a, b = c
        return b, b * 2.0 + 1.0

    return lax.fori_loop(0, _dyn8(x), body, (x, x + 3.0))


_CASES = {
    "forward_up_dynamic": (forward_up_dynamic, lambda: np.arange(4, dtype=np.float32)),
    "forward_up_through_add_zero": (forward_up_through_add_zero, lambda: np.arange(4, dtype=np.float32)),
    "forward_down_dynamic": (forward_down_dynamic, lambda: np.arange(4, dtype=np.float32)),
    "fib_dynamic_f32": (fib_dynamic, lambda: np.int32(20)),
    "fib_dynamic_i32": (fib_dynamic_int, lambda: np.int32(25)),
    "shift_with_row_write": (
        shift_with_row_write,
        lambda: np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32),
    ),
    "three_cycle": (three_cycle, lambda: np.arange(6, dtype=np.float32)),
    "while_swap": (while_swap, lambda: np.float32(1.0)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_carry_forwarding_matches_cpu(case):
    fn, make = _CASES[case]
    x = make()
    want = _on("cpu", fn, x)
    got = _on("mps", fn, x)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
        assert g.dtype == w.dtype and g.shape == w.shape
        np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-6)
