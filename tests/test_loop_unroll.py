# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Counted-loop unrolling (JAX_METALLIB_LOOP_UNROLL) must not change results.

The unroll factor is read once per process, so every case runs in a child
process with the setting on and is compared against XLA:CPU in that child.
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

    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)

    def fori(lo, hi):
        def f(a):
            return lax.fori_loop(lo, hi, lambda i, c: jnp.tanh(c * 0.9 + a[i % 16] * i.astype(a.dtype)), a[0])
        return f

    def strided_while(lo, hi, step):
        def f(a):
            def cond(s):
                return s[0] < hi
            def body(s):
                i, c = s
                return i + step, c + a[i % 16] * (i + 1).astype(a.dtype)
            return lax.while_loop(cond, body, (jnp.int32(lo), jnp.zeros((32,), a.dtype)))[1]
        return f

    def scan_carry_only(a):
        def step(c, t):
            return (c[0] + a[t % 16].sum() * t, c[1] * 1.001 + 1.0), None
        (s, p), _ = lax.scan(step, (jnp.float32(0), jnp.float32(1)), jnp.arange(96, dtype=jnp.int32))
        return jnp.stack([s, p])

    def int_carry(a):
        return lax.fori_loop(0, 64, lambda i, c: c * 3 + i, jnp.int32(7))

    def read_only(a):
        def step(c, t):
            return c + lax.dynamic_slice(a, (t % 16, 0), (1, 32)).sum(), None
        return lax.scan(step, jnp.float32(0), jnp.arange(128, dtype=jnp.int32))[0]

    CASES = {
        "fori_divisible": fori(0, 64),
        "fori_prime": fori(0, 61),
        "fori_lower_bound": fori(5, 69),
        "fori_short": fori(0, 3),
        "while_step3_uneven": strided_while(1, 50, 3),
        "while_step2": strided_while(0, 64, 2),
        "while_step3_span47": strided_while(2, 49, 3),
        "scan_carry_only": scan_carry_only,
        "int_carry_wraps": int_carry,
        "read_only": read_only,
    }
    fn = CASES[sys.argv[1]]
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    print("OK")
    """
)

_CASES = [
    "fori_divisible",
    "fori_prime",
    "fori_lower_bound",
    "fori_short",
    "while_step3_uneven",
    "while_step2",
    "while_step3_span47",
    "scan_carry_only",
    "int_carry_wraps",
    "read_only",
]


@pytest.mark.parametrize("unroll", ["1", "4", "8"])
@pytest.mark.parametrize("case", _CASES)
def test_counted_loop_unroll_matches_cpu(case, unroll):
    env = {**os.environ, "JAX_PLATFORMS": "cpu,mps", "JAX_METALLIB_LOOP_UNROLL": unroll}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, case], capture_output=True, text=True, timeout=120, env=env, check=False
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
