# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""reduce_window max/min fold structures match XLA:CPU bitwise.

The shared extrema slice fold (integer windows and the float order-key path)
folds one axis at a time for larger or strided multi-axis windows and by
doubling on stride-1 undilated axes of >= 32 positions
(JAX_METALLIB_DISABLE_WINDOW_SEPARABLE_FOLD=1 keeps one slice per position).
Both rewrites are exact; these cases pin padding, strides, dilation and
non-power-of-two doubling windows. Shapes avoid XLA:CPU's 2x2 stride-2 pooling
artefact (>= 64x64).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32}


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _bits(a):
    a = np.asarray(a)
    return a.reshape(-1).view(UINT[a.dtype.itemsize])


def _check(fn, x):
    cpu, mps = _devices()
    want = np.asarray(jax.jit(fn)(jax.device_put(x, cpu)))
    got = np.asarray(jax.jit(fn)(jax.device_put(x, mps)))
    assert want.shape == got.shape and want.dtype == got.dtype
    bad = np.flatnonzero(_bits(want) != _bits(got))
    assert bad.size == 0, (
        f"{bad.size} mismatches, first at {bad[:5]}: cpu={want.reshape(-1)[bad[:5]]} mps={got.reshape(-1)[bad[:5]]}"
    )


# name: (window, strides, padding, window_dilation, shape)
CASES = {
    "dbl_1x40_same": ((1, 40), (1, 1), [(0, 0), (19, 20)], (1, 1), (8, 96)),
    "dbl_32x1_same": ((32, 1), (1, 1), [(15, 16), (0, 0)], (1, 1), (48, 16)),
    "dbl_1x33_valid": ((1, 33), (1, 1), [(0, 0), (0, 0)], (1, 1), (6, 80)),
    "dbl_2x32_pad": ((2, 32), (1, 1), [(1, 0), (3, 5)], (1, 1), (10, 70)),
    "sep_2x2s2": ((2, 2), (2, 2), [(0, 0), (0, 0)], (1, 1), (32, 32)),
    "sep_3x3s2_pad": ((3, 3), (2, 2), [(1, 1), (0, 1)], (1, 1), (31, 30)),
    "sep_5x5s1": ((5, 5), (1, 1), [(2, 2), (2, 2)], (1, 1), (24, 28)),
    "sep_4x6s3x2_pad": ((4, 6), (3, 2), [(1, 2), (0, 3)], (1, 1), (29, 31)),
    "sep_1x8s8": ((1, 8), (1, 8), [(0, 0), (0, 0)], (1, 1), (12, 64)),
    "dil_1x20d2": ((1, 20), (1, 1), [(0, 0), (19, 19)], (1, 2), (6, 64)),
    "dil_3x3d2s2": ((3, 3), (2, 2), [(2, 2), (2, 2)], (2, 2), (20, 22)),
}


def _fn(win, strides, pad, dil, init, op):
    return lambda x: lax.reduce_window(x, init, op, win, strides, pad, window_dilation=dil)


def _float_input(dt, shape, seed):
    vals = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 2.5, -3.0, 0.5], np.float32)
    rng = np.random.default_rng(seed)
    x = rng.choice(vals, size=shape)
    # Mostly finite random values so most windows are decided by ordering.
    mask = rng.random(shape) < 0.7
    x[mask] = rng.standard_normal(mask.sum()).astype(np.float32)
    return x.astype(dt)


@pytest.mark.parametrize("dtname", ["f16", "bf16"])
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("case", list(CASES))
def test_float_key_fold_structure(dtname, kind, case):
    dt = {"f16": jnp.float16, "bf16": jnp.bfloat16}[dtname]
    win, strides, pad, dil, shape = CASES[case]
    d = jnp.dtype(dt)
    init, op = (d.type(-np.inf), lax.max) if kind == "max" else (d.type(np.inf), lax.min)
    _check(_fn(win, strides, pad, dil, init, op), _float_input(dt, shape, seed=len(case)))


@pytest.mark.parametrize("dt", [np.int32, np.int8, np.uint16])
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("case", list(CASES))
def test_integer_fold_structure(dt, kind, case):
    win, strides, pad, dil, shape = CASES[case]
    info = np.iinfo(dt)
    rng = np.random.default_rng(len(case))
    x = rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dt)
    init, op = (dt(info.min), lax.max) if kind == "max" else (dt(info.max), lax.min)
    _check(_fn(win, strides, pad, dil, init, op), x)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_float_key_fold_inside_scan(kind):
    # Doubling and separable folds inside a loop body.
    op, init = (lax.max, np.float16(-np.inf)) if kind == "max" else (lax.min, np.float16(np.inf))

    def body(c, _):
        y = lax.reduce_window(c, init, op, (1, 40), (1, 1), [(0, 0), (19, 20)])
        z = lax.reduce_window(y, init, op, (3, 3), (1, 1), [(1, 1), (1, 1)])
        return z * np.float16(0.5) + c, None

    _check(lambda x: lax.scan(body, x, None, length=3)[0], _float_input(jnp.float16, (8, 96), seed=7))
