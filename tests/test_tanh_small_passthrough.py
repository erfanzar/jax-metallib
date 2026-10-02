"""f32 tanh passthrough cutoff matches XLA (|x| < 4e-4 -> x).

HandleTanh used 2e-4, so [2e-4, 4e-4) went through MPS tanh, which cancels there
(0.05% bitwise vs XLA:CPU, up to 3441 ulp). With XLA's 4e-4 cutoff the whole
|x| < 4e-4 range is bitwise equal to XLA:CPU. MPS tanh above 4e-4 is not bitwise
(about 37% on |x| >= 0.5, max ~5 ulp; no FMA in MPSGraph), so only the small
range is asserted here. Runs in a child so JAX_PLATFORMS is set for it alone.
"""

import os
import subprocess
import sys
import textwrap

import pytest

CHILD = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    cpu = jax.devices("cpu")[0]
    mps = jax.devices("mps")[0]
    rng = np.random.default_rng(0)
    band = rng.uniform(2e-4, 4e-4, 1 << 16)
    tiny = np.logspace(-45, np.log10(3.999e-4), 1 << 14)
    edges = np.array([0.0, 1e-45, 1e-38, 1e-30, 1.999e-4, 2e-4, 2.001e-4, 3.999e-4, np.nextafter(np.float32(4e-4), 0)])
    xs = np.concatenate([edges, -edges, band, -band, tiny, -tiny]).astype(np.float32)
    assert np.all(np.abs(xs) < np.float32(4e-4))
    f = jax.jit(jnp.tanh)
    got = np.asarray(f(jax.device_put(xs, mps))).view(np.uint32)
    want = np.asarray(f(jax.device_put(xs, cpu))).view(np.uint32)
    bad = np.flatnonzero(got != want)
    print(f"checked={xs.size} mismatches={bad.size} first={xs[bad[:5]].tolist()}", flush=True)
    assert bad.size == 0
    assert np.array_equal(got, xs.view(np.uint32))
    nan = np.array([np.nan, -np.nan], np.float32)
    assert np.all(np.isnan(np.asarray(f(jax.device_put(nan, mps)))))
    print("TANH_SMALL_OK", flush=True)
    """
)


def test_tanh_small_passthrough_matches_xla():
    import jax

    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    proc = subprocess.run(
        [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TANH_SMALL_OK" in proc.stdout
