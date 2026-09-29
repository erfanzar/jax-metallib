# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sorting complex values must match XLA CPU and must not abort on MPS.

mps.sort / argSort reject complex operands with a process-aborting assertion
("'mps.sort' op operand #0 must be tensor of mps native type values"), so
jnp.sort, jnp.argsort and lax.sort with complex keys killed the process. JAX
orders complex values lexicographically by (real, imag). Each case runs in a
child process so an abort is reported as a test failure instead of killing
pytest.
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
    rng = np.random.default_rng(5)
    # Real parts drawn from a small set so ties force the imaginary tie-break.
    re = rng.choice(np.array([-2.0, -0.5, 0.0, 1.0, 3.0], np.float32), size=(4, 9))
    im = rng.standard_normal((4, 9)).astype(np.float32)
    z = (re + 1j * im).astype(np.complex64)
    vals = rng.standard_normal((4, 9)).astype(np.float32)
    cases = {
        "sort_1d": (lambda a: jnp.sort(a), (z[0],)),
        "sort_axis0": (lambda a: jnp.sort(a, axis=0), (z,)),
        "sort_axis1": (lambda a: jnp.sort(a, axis=1), (z,)),
        "sort_descending": (lambda a: jnp.sort(a, axis=1, descending=True), (z,)),
        "argsort": (lambda a: jnp.argsort(a, axis=1), (z,)),
        "sort_key_value": (lambda a, b: lax.sort((a, b), dimension=1, num_keys=1), (z, vals)),
    }
    fn, args = cases[name]
    f = jax.jit(fn) if jit else fn
    out = {}
    for p in ("cpu", "mps"):
        d = jax.devices(p)[0]
        with jax.default_device(d):
            res = f(*[jax.device_put(a, d) for a in args])
            out[p] = [np.asarray(r) for r in jax.tree_util.tree_leaves(res)]
    for c, m in zip(out["cpu"], out["mps"], strict=True):
        assert c.dtype == m.dtype and c.shape == m.shape, (c.dtype, m.dtype, c.shape, m.shape)
        assert np.array_equal(c, m), (c, m)
    print("OK")
    """
)

_CASES = ["sort_1d", "sort_axis0", "sort_axis1", "sort_descending", "argsort", "sort_key_value"]


@pytest.mark.parametrize("jit", ["jit", "eager"])
@pytest.mark.parametrize("case", _CASES)
def test_complex_sort_matches_cpu(case, jit):
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
