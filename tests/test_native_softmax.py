# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Last-axis softmax under jit must match XLA:CPU, including NaN / inf rows.

jax.nn.softmax lowers to reduce_max -> sub -> exp -> reduce_sum -> div, which
the plugin can fuse into one native kernel. Each case also runs in a child
process with the intrinsic kernel family forced on (the flag is read once).
"""

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_RNG = np.random.default_rng(7)


def _special(shape, dtype=np.float32):
    x = (_RNG.standard_normal(shape) * 4).astype(np.float32)
    x[0, 3] = np.nan
    x[1, :] = -np.inf
    x[2, 5] = np.inf
    x[3, 0] = np.nan
    x[3, 1:] = -np.inf
    x[4, :] = 80.0
    x[5, 7] = -np.inf
    return x.astype(dtype)


_CASES = {
    "f32_4096x64": lambda: _RNG.standard_normal((4096, 64)).astype(np.float32),
    "f32_256x4096": lambda: _RNG.standard_normal((256, 4096)).astype(np.float32),
    "f32_16x10000_two_pass": lambda: _RNG.standard_normal((16, 10000)).astype(np.float32),
    "f32_rank3_4x128x300": lambda: _RNG.standard_normal((4, 128, 300)).astype(np.float32),
    "f32_special_64x1000": lambda: _special((64, 1000)),
    "f32_special_8x9000": lambda: _special((8, 9000)),
    "f16_512x1024": lambda: _RNG.standard_normal((512, 1024)).astype(np.float16),
    "f16_special_64x1000": lambda: _special((64, 1000), np.float16),
}


def _softmax(a):
    return jax.nn.softmax(a, axis=-1)


def _grad(a):
    return jax.grad(lambda v: jnp.sum(jax.nn.softmax(v, axis=-1) * jnp.arange(v.shape[-1], dtype=v.dtype)))(a)


def _check(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    tol = 3e-3 if expected.dtype == np.float16 else 2e-5
    np.testing.assert_allclose(
        np.nan_to_num(actual.astype(np.float32)), np.nan_to_num(expected.astype(np.float32)), rtol=tol, atol=tol
    )


def _run(fn, x):
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    return actual, expected


@pytest.mark.parametrize("case", list(_CASES))
def test_softmax_matches_cpu(case):
    _check(*_run(_softmax, _CASES[case]()))


@pytest.mark.parametrize("case", ["f32_4096x64", "f32_rank3_4x128x300"])
def test_softmax_grad_matches_cpu(case):
    _check(*_run(_grad, _CASES[case]()))


_CHILD = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[3])
    import test_native_softmax as t
    fn = t._grad if sys.argv[2] == "grad" else t._softmax
    t._check(*t._run(fn, t._CASES[sys.argv[1]]()))
    print("OK")
    """
)


@pytest.mark.parametrize("kind", ["value", "grad"])
@pytest.mark.parametrize("case", list(_CASES))
def test_softmax_matches_cpu_with_intrinsic_kernels(case, kind):
    if kind == "grad" and case not in ("f32_4096x64", "f32_rank3_4x128x300"):
        pytest.skip("gradient checked on two representative shapes")
    env = {**os.environ, "JAX_PLATFORMS": "cpu,mps", "JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, case, kind, os.path.dirname(__file__)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
