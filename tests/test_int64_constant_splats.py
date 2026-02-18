import os
import subprocess
import sys
import textwrap

import jax
import pytest


def test_int64_constant_splat_matches_cpu():
    if not jax.devices("mps"):
        pytest.skip("MPS device not available")

    code = textwrap.dedent(
        r"""
        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", True)

        # Value above 2**53 to catch precision loss if routed through double.
        val = 2**53 + 1

        def f():
            return jnp.full((1024,), val, dtype=jnp.int64)

        cpu = jax.jit(f, backend="cpu")()
        mps = jax.jit(f, backend="mps")()

        assert cpu.device.platform == "cpu", cpu.device
        assert mps.device.platform == "mps", mps.device

        cpu_np = np.asarray(cpu)
        mps_np = np.asarray(mps)

        assert cpu_np.dtype == np.int64
        assert mps_np.dtype == np.int64
        assert cpu_np[0] == val, cpu_np[0]
        assert mps_np[0] == val, mps_np[0]
        assert np.array_equal(cpu_np, mps_np)

        print("ok")
        """
    )

    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

