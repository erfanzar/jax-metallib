# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""schur / eigvals at n >= 64 on MPS must not abort the process.

The shifted-QR sweep of the Schur lowerings ran inside lax.while_loop
(MPSGraph whileWithInitialInputs). From n = 64 up, MPSGraph's
MPSGPURegionFormation pass failed on that body ('operand #0 does not dominate
this use') and aborted the process with SIGABRT. Each case runs in a
subprocess so a regression shows up as a failed test, not a dead pytest run.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
    import numpy as np

    case, n = sys.argv[1], int(sys.argv[2])
    rng = np.random.default_rng(n)
    x = rng.standard_normal((n, n)).astype(np.float32)
    mps = jax.devices("mps")[0]
    xd = jax.device_put(x, mps)
    if case == "eigvals":
        w = np.asarray(jax.jit(jnp.linalg.eigvals)(xd))
        ref = np.linalg.eigvals(x.astype(np.float64))
        d = np.abs(w[:, None] - ref[None, :])
        err = max(d.min(axis=1).max(), d.min(axis=0).max())
        assert err <= 1e-2 * max(1.0, np.abs(ref).max()), err
    else:
        output = "complex" if case == "schur_complex" else "real"
        t, z = jax.jit(lambda a: jsl.schur(a, output=output))(xd)
        t, z = np.asarray(t), np.asarray(z)
        zh = np.conj(z.T)
        np.testing.assert_allclose(zh @ z, np.eye(n), atol=1e-3)
        np.testing.assert_allclose(z @ t @ zh, x, atol=5e-3 * np.abs(x).max() * np.sqrt(n))
        lim = -2 if output == "real" else -1
        assert np.all(np.tril(t, lim) == 0) or np.abs(np.tril(t, lim)).max() <= 1e-3
    print("OK")
    """
)


@pytest.mark.parametrize("n", [64, 96])
@pytest.mark.parametrize("case", ["schur_real", "schur_complex", "eigvals"])
def test_large_schur_does_not_abort(case, n):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, case, str(n)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-2000:]
    assert proc.returncode == 0, f"exit {proc.returncode}:\n{tail}"
    assert "OK" in proc.stdout, tail
