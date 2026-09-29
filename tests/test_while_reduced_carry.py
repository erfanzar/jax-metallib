# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""lax.while_loop whose cond reads a reduced carry must not crash on MPS.

MPSGraph whileWithInitialInputs segfaulted when a carry read by the predicate
(and produced by a reduction in the body) was not a compile target, e.g. a
convergence loop testing max|a| whose jitted function returned only `a`.
HandleWhileOp now ties every loop output to a NaN-safe zero that depends on all
outputs. Each case runs in a child process so an abort is reported as a test
failure instead of killing pytest.
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
    x = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)

    def shrink_loop(a, *, return_all):
        def cond(s):
            k, _, m = s
            return (k < 12) & (m > 1e-4)

        def body(s):
            k, a, _ = s
            a = a * 0.99 + 0.001
            return k + 1, a, jnp.max(jnp.abs(a))

        out = lax.while_loop(cond, body, (jnp.int32(0), a, jnp.float32(1.0)))
        return out if return_all else out[1]

    def int_sum_carry(a):
        def body(s):
            k, a, _ = s
            a = a * 0.5
            return k + 1, a, (jnp.sum(jnp.abs(a)) * 100).astype(jnp.int32)

        return lax.while_loop(lambda s: (s[0] < 20) & (s[2] > 3), body, (jnp.int32(0), a, jnp.int32(1 << 20)))[1]

    def bool_carry(a):
        def body(s):
            k, a, _ = s
            a = a * 0.5
            return k + 1, a, jnp.any(jnp.abs(a) > 0.1)

        return lax.while_loop(lambda s: (s[0] < 30) & s[2], body, (jnp.int32(0), a, jnp.bool_(True)))[1]

    def newton_sqrt(c):
        def body(s):
            k, y, _ = s
            y2 = 0.5 * (y + c / y)
            return k + 1, y2, jnp.max(jnp.abs(y2 - y))

        out = lax.while_loop(lambda s: (s[0] < 50) & (s[2] > 1e-6), body, (jnp.int32(0), c, jnp.float32(1.0)))
        return out[1]

    def keeps_special_values(a):
        # -0 and NaN entries must pass through the loop unchanged in sign and NaN-ness.
        def body(s):
            k, a, _ = s
            a = a * 1.0
            return k + 1, a, jnp.max(jnp.abs(jnp.nan_to_num(a)))

        return lax.while_loop(lambda s: (s[0] < 3) & (s[2] > 0), body, (jnp.int32(0), a, jnp.float32(1.0)))[1]

    special = x.copy()
    special[0, :4] = [-0.0, np.nan, np.inf, -np.inf]
    cases = {
        "reduced_carry_return_a": (lambda a: shrink_loop(a, return_all=False), x),
        "reduced_carry_return_all": (lambda a: shrink_loop(a, return_all=True), x),
        "int_sum_carry": (int_sum_carry, x),
        "bool_carry": (bool_carry, x),
        "newton_sqrt": (newton_sqrt, np.abs(x) + 1.0),
        "keeps_special_values": (keeps_special_values, special),
    }
    fn, arg = cases[name]
    f = jax.jit(fn) if jit else fn
    out = {}
    for p in ("cpu", "mps"):
        d = jax.devices(p)[0]
        with jax.default_device(d):
            for _ in range(20 if p == "mps" else 1):
                res = jax.block_until_ready(f(jax.device_put(arg, d)))
            out[p] = [np.asarray(r) for r in jax.tree_util.tree_leaves(res)]
    for c, m in zip(out["cpu"], out["mps"], strict=True):
        assert c.dtype == m.dtype and c.shape == m.shape, (c.dtype, m.dtype, c.shape, m.shape)
        if np.issubdtype(c.dtype, np.floating):
            np.testing.assert_allclose(m, c, rtol=1e-5, atol=1e-6, equal_nan=True)
            zero = c == 0
            assert np.array_equal(np.signbit(m[zero]), np.signbit(c[zero])), "signed zero changed"
        else:
            np.testing.assert_array_equal(m, c)
    print("OK")
    """
)

_CASES = [
    "reduced_carry_return_a",
    "reduced_carry_return_all",
    "int_sum_carry",
    "bool_carry",
    "newton_sqrt",
    "keeps_special_values",
]


@pytest.mark.parametrize("jit", ["jit", "eager"])
@pytest.mark.parametrize("case", _CASES)
def test_while_reduced_carry_matches_cpu(case, jit):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, case, jit],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        check=False,
    )
    assert r.returncode == 0 and "OK" in r.stdout, (r.returncode, r.stderr[-800:])
