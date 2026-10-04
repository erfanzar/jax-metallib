# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Row-wise float32 sort must match XLA:CPU ordering, including NaN and signed zero.

Shapes here (rows > 1, cols > 256) are eligible for the native bitonic kernel.
XLA sorts NaN after +inf. The native path is also exercised with the intrinsic
kernel family forced on, in a child process, because that flag is read once.
"""

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_RNG = np.random.default_rng(3)


def _special(shape):
    x = _RNG.standard_normal(shape).astype(np.float32)
    flat = x.reshape(-1)
    idx = _RNG.choice(flat.size, size=flat.size // 8, replace=False)
    specials = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], np.float32)
    flat[idx] = specials[np.arange(idx.size) % specials.size]
    return x


_CASES = {
    "normal_512x300": lambda: _RNG.standard_normal((512, 300)).astype(np.float32),
    "special_512x300": lambda: _special((512, 300)),
    "special_64x1000": lambda: _special((64, 1000)),
    "special_8x4096": lambda: _special((8, 4096)),
    "all_nan_rows_16x512": lambda: np.full((16, 512), np.nan, np.float32),
    "duplicates_128x513": lambda: _RNG.integers(-3, 4, (128, 513)).astype(np.float32),
    "benchmark_4096x256": lambda: _RNG.standard_normal((4096, 256)).astype(np.float32),
}


def _sort(a):
    return jax.lax.sort(a, dimension=-1)


def _check(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.nan_to_num(actual), np.nan_to_num(expected))


@pytest.mark.parametrize("case", list(_CASES))
def test_row_sort_matches_cpu(case):
    x = _CASES[case]()
    expected = np.asarray(jax.jit(_sort, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(_sort)(jnp.asarray(x)))
    _check(actual, expected)


@pytest.mark.parametrize("case", ["special_512x300", "duplicates_128x513", "special_8x4096"])
def test_argsort_matches_cpu(case):
    # Indices expose ordering bugs that value comparisons hide (NaN sign, -0 ties).
    x = _CASES[case]()
    expected = np.asarray(jax.jit(lambda a: jnp.argsort(a, axis=-1), backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(lambda a: jnp.argsort(a, axis=-1))(jnp.asarray(x)))
    np.testing.assert_array_equal(actual, expected)


_CHILD = textwrap.dedent(
    """
    import sys
    import jax
    import numpy as np
    from jax import numpy as jnp
    sys.path.insert(0, sys.argv[2])
    import test_native_sort as t
    x = t._CASES[sys.argv[1]]()
    expected = np.asarray(jax.jit(t._sort, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(t._sort)(jnp.asarray(x)))
    t._check(actual, expected)
    print("OK")
    """
)


@pytest.mark.parametrize("case", list(_CASES))
def test_row_sort_matches_cpu_with_intrinsic_kernels(case):
    env = {**os.environ, "JAX_PLATFORMS": "cpu,mps", "JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, case, os.path.dirname(__file__)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
