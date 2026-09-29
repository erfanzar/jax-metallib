# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""lax.rng_uniform (stablehlo.rng UNIFORM) must not abort the process on MPS.

MPSGraph randomTensorWithShape: returns a tensor whose static shape is -1, and
the plugin sized the result from it, aborting with "MPSNDArray ... dimension
length > INT_MAX". Each case runs in a child process so an abort is reported as
a test failure instead of killing pytest.
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

    lo, hi, shape = float(sys.argv[1]), float(sys.argv[2]), eval(sys.argv[3])
    jit, runtime = sys.argv[4] == "jit", sys.argv[5] == "runtime"
    with jax.default_device(jax.devices("mps")[0]):
        if runtime:
            fn = lambda a, b: lax.rng_uniform(a, b, shape)
            args = (jnp.float32(lo), jnp.float32(hi))
        else:
            fn = lambda: lax.rng_uniform(jnp.float32(lo), jnp.float32(hi), shape)
            args = ()
        out = np.asarray((jax.jit(fn) if jit else fn)(*args))
    assert out.shape == shape, out.shape
    assert out.dtype == np.float32, out.dtype
    assert np.all(out >= lo) and np.all(out < hi), out
    print("OK")
    """
)


@pytest.mark.parametrize("jit", ["jit", "eager"])
@pytest.mark.parametrize("bounds", ["constant", "runtime"])
@pytest.mark.parametrize(
    ("lo", "hi", "shape"), [(0.0, 1.0, (4,)), (2.0, 5.0, (3, 5)), (-1.0, 1.0, (2, 3, 4))], ids=["1d", "2d", "3d"]
)
def test_rng_uniform_runs_and_respects_bounds(lo, hi, shape, bounds, jit):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, str(lo), str(hi), repr(shape), jit, bounds],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert r.returncode == 0 and "OK" in r.stdout, (r.returncode, r.stderr[-600:])
