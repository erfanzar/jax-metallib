# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""A completion timeout must never kill the process.

CompletionEvent::Await() throws std::runtime_error after
JAX_METALLIB_COMPLETION_TIMEOUT_MS. The detached CacheReleaseWorker thread
waits on each execution's fence before freeing its intermediate buffers; a
throw there is std::terminate (SIGABRT) whatever the caller does, so any GPU
program slower than the timeout (e.g. complex schur at n=256, ~47 s) killed
the process. Release waits must wait for the GPU instead; user-facing waits may
report the timeout as an error. Each case runs in a subprocess so a regression
is a failed test, not a dead pytest run.
"""

import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import os
    import jax
    import jax.scipy.linalg as jsl
    import numpy as np

    n = 96
    x = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    mps = jax.devices("mps")[0]
    xd = jax.device_put(x, mps)
    f = jax.jit(lambda a: jsl.schur(a, output="complex"))

    os.environ["JAX_METALLIB_COMPLETION_TIMEOUT_MS"] = "1000"
    outcome = "finished"
    try:
        np.asarray(f(xd)[0])
    except Exception as e:  # noqa: BLE001
        outcome = "raised" if "timed out" in str(e) else "other: " + repr(e)[:300]
    os.environ["JAX_METALLIB_COMPLETION_TIMEOUT_MS"] = "120000"

    t, z = (np.asarray(v) for v in f(xd))
    zh = np.conj(z.T)
    np.testing.assert_allclose(zh @ z, np.eye(n), atol=1e-3)
    np.testing.assert_allclose(z @ t @ zh, x, atol=5e-3 * np.abs(x).max() * np.sqrt(n))
    print("OUTCOME", outcome)
    """
)


def test_completion_timeout_does_not_abort():
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    env.pop("JAX_METALLIB_COMPLETION_TIMEOUT_MS", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-2000:]
    assert proc.returncode == 0, f"exit {proc.returncode}:\n{tail}"
    assert "OUTCOME" in proc.stdout, tail
    outcome = proc.stdout.split("OUTCOME", 1)[1].strip()
    assert outcome in ("raised", "finished"), tail
