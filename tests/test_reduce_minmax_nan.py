# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""max/min reductions must propagate NaN (StableHLO maximum/minimum do).

The native TG/linear/axis-0 max/min kernels used Metal max/min and
simd_max/simd_min, which return the non-NaN operand, so jnp.max/jnp.min over
>= 4096 elements silently dropped NaN (f32 and f16, jit and eager). The
reference is NumPy, not XLA:CPU: jaxlib 0.11.1's CPU backend also drops NaN
here (f32 max over 4096 elements with one NaN returns -inf), so a CPU-compare
test cannot catch this.
"""

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_SIZES = (1000, 4096, 65537, 1 << 20)
_DTYPES = (np.float32, np.float16)

_FNS = {
    "max": (jnp.max, np.max),
    "min": (jnp.min, np.min),
    "max_x2": (lambda v: jnp.max(v * 2), lambda v: np.max(v * np.asarray(2, v.dtype))),
    "min_2d": (lambda v: jnp.min(v.reshape(-1, 8)), lambda v: np.min(v.reshape(-1, 8))),
    "max_axis1_row8": (
        lambda v: jnp.max(v.reshape(-1, 8), axis=1),
        lambda v: np.max(v.reshape(-1, 8), axis=1),
    ),
    "min_axis0_col8": (
        lambda v: jnp.min(v.reshape(-1, 8), axis=0),
        lambda v: np.min(v.reshape(-1, 8), axis=0),
    ),
}


def _make(n, dtype, fill):
    x = np.random.default_rng(n).standard_normal(n).astype(dtype)
    if fill == "nan_first":
        x[0] = np.nan
    elif fill == "nan_mid":
        x[n // 2] = np.nan
    elif fill == "nan_last":
        x[n - 1] = np.nan
    elif fill == "inf_only":
        x[n // 3] = np.inf
        x[7] = -np.inf
    elif fill == "all_nan":
        x[:] = np.nan
    return x


def _run(fn, x):
    with jax.default_device(jax.devices("mps")[0]):
        return np.asarray(jax.jit(fn)(jnp.asarray(x)))


@pytest.mark.parametrize("fill", ["nan_first", "nan_mid", "nan_last", "inf_only", "all_nan"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("name", list(_FNS))
def test_minmax_reduce_propagates_nan(name, n, dtype, fill):
    if name in ("min_2d", "max_axis1_row8", "min_axis0_col8") and n % 8:
        pytest.skip("reshape needs a multiple of 8")
    fn, ref = _FNS[name]
    x = _make(n, dtype, fill)
    actual = _run(fn, x)
    expected = ref(x)
    assert actual.dtype == expected.dtype and actual.shape == np.shape(expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
def test_minmax_reduce_eager_propagates_nan(dtype):
    x = _make(65536, dtype, "nan_mid")
    with jax.default_device(jax.devices("mps")[0]):
        xd = jnp.asarray(x)
        assert np.isnan(np.asarray(jnp.max(xd)))
        assert np.isnan(np.asarray(jnp.min(xd)))


_INTRINSIC_SCRIPT = textwrap.dedent(
    """
    import jax, jax.numpy as jnp, numpy as np
    bad = []
    numer = jax.jit(lambda v: jnp.exp(v - jnp.max(v, axis=-1, keepdims=True)))
    for shape in ((8, 4096), (1, 4096), (16, 1024)):
        for dt in (np.float32, np.float16):
            x = np.random.default_rng(0).standard_normal(shape).astype(dt)
            x[0, shape[1] // 3] = np.nan
            got = np.asarray(numer(jnp.asarray(x)))
            if not np.isnan(got[0]).all() or np.isnan(got[1:]).any():
                bad.append(('numer', shape, np.dtype(dt).name))
            for name, fn in (('max', jnp.max), ('min', jnp.min)):
                if not np.isnan(np.asarray(jax.jit(fn)(jnp.asarray(x)))):
                    bad.append((name, shape, np.dtype(dt).name))
    print('BAD', bad) if bad else print('OK')
    """
)


def test_intrinsic_softmax_numerator_and_reduce_propagate_nan():
    # The softmax_numerator_* kernels are only reached with the opt-in
    # intrinsic family, which is read once per process, so run in a child.
    # JAX_PLATFORMS is overridden (not setdefault): the suite exports mps.
    env = dict(os.environ, JAX_PLATFORMS="mps", JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS="1")
    proc = subprocess.run(
        [sys.executable, "-c", _INTRINSIC_SCRIPT], env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("OK"), proc.stdout[-2000:]


_INTRINSIC_BINARY_SCRIPT = textwrap.dedent(
    """
    import jax, jax.numpy as jnp, numpy as np
    bad = []
    # 17 -> scalar kernel; 4096 -> vec4/vec8; 1<<20, 1<<22 -> vec8/stride.
    for n in (17, 4096, 1 << 20, 1 << 22):
        for dt in (np.float32, np.float16):
            rng = np.random.default_rng(n)
            for where in ('a', 'b', 'both'):
                a = rng.standard_normal(n).astype(dt)
                b = rng.standard_normal(n).astype(dt)
                idx = rng.choice(n, size=max(1, n // 7), replace=False)
                if where in ('a', 'both'):
                    a[idx] = np.nan
                if where in ('b', 'both'):
                    b[idx[::2]] = np.nan
                for name, fn, ref in (('max', jnp.maximum, np.maximum), ('min', jnp.minimum, np.minimum)):
                    got = np.asarray(jax.jit(fn)(jnp.asarray(a), jnp.asarray(b)))
                    want = ref(a, b)
                    gn, wn = np.isnan(got), np.isnan(want)
                    if not np.array_equal(gn, wn) or not np.array_equal(got[~gn], want[~wn]):
                        bad.append((name, n, np.dtype(dt).name, where))
    print('BAD', bad) if bad else print('OK')
    """
)


def test_intrinsic_elementwise_max_min_propagate_nan():
    # The native binary_{max,min}_{,vec4_,vec8_,stride_}{f32,f16} kernels used Metal
    # max/min, which return the non-NaN operand. They are only reached with the
    # opt-in intrinsic family; autotune is disabled so the ops are not demoted
    # to MPSGraph. Single ops only: fused chains use a codegen template outside
    # these kernels.
    env = dict(
        os.environ,
        JAX_PLATFORMS="mps",
        JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS="1",
        JAX_METALLIB_DISABLE_ELEMWISE_AUTOTUNE="1",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _INTRINSIC_BINARY_SCRIPT], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("OK"), proc.stdout[-2000:]
