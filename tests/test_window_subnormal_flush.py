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
"""f32 window max/min flushes subnormals to a signed zero, as XLA:CPU does.

Pooling, cumulative and slice-fold paths outside the float order-key gate used to
return the subnormal itself (ref=+-0, got=+-1e-40). The reference is NumPy: flush
f32 subnormals to a signed zero, order -0 < +0, propagate NaN as canonical qNaN.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from numpy.lib.stride_tricks import sliding_window_view


def _mps():
    try:
        return jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")


def _ref(x, is_max, dims, strides, pads):
    b = x.view(np.int32).astype(np.int64)
    b = np.where((b & 0x7F800000) == 0, np.where(b < 0, -(2**31), 0), b)
    key = np.where(b < 0, b ^ 0x7FFFFFFF, b)
    nan_key = 2**31 - 1 if is_max else -(2**31)
    key = np.where((b & 0x7FFFFFFF) > 0x7F800000, nan_key, key)
    padded = np.pad(key, pads, constant_values=-(2**31) if is_max else 2**31 - 1)
    win = sliding_window_view(padded, dims)[tuple(slice(None, None, s) for s in strides)]
    axes = tuple(range(x.ndim, 2 * x.ndim))
    r = win.max(axis=axes) if is_max else win.min(axis=axes)
    bits = np.where(r < 0, r ^ 0x7FFFFFFF, r)
    bits = np.where(r == nan_key, 0x7FC00000, bits)
    return bits.astype(np.int32).view(np.float32)


# name: (is_max, dims, strides, padding, shape)
WINDOWS = {
    "pool3x3_s2_same": (True, (3, 3), (2, 2), "SAME", (32, 32)),
    "pool3x3_s2_valid": (False, (3, 3), (2, 2), "VALID", (32, 32)),
    "pool4x4_s4": (True, (4, 4), (4, 4), "VALID", (32, 32)),
    "pool8x8_s8": (False, (8, 8), (8, 8), "VALID", (32, 32)),
    "slide1x8_s8": (True, (1, 8), (1, 8), "VALID", (32, 32)),
    "slide8x1_s8": (False, (8, 1), (8, 1), "VALID", (32, 32)),
    "nhwc_max3s2": (True, (1, 3, 3, 1), (1, 2, 2, 1), "SAME", (2, 16, 16, 4)),
    "nhwc_min2s2": (False, (1, 2, 2, 1), (1, 2, 2, 1), "VALID", (2, 16, 16, 4)),
    "nhwc_max3s1": (True, (1, 3, 3, 1), (1, 1, 1, 1), "SAME", (2, 16, 16, 4)),
    "rank3_max2": (True, (2, 2, 2), (1, 1, 1), "VALID", (4, 16, 16)),
    "rank3_min2s2": (False, (2, 2, 2), (2, 2, 2), "VALID", (4, 16, 16)),
    # Inside the float order-key gate (already flushed there): controls.
    "slide1x3_s1": (True, (1, 3), (1, 1), "SAME", (32, 32)),
    "pool5x5_s1": (False, (5, 5), (1, 1), "SAME", (32, 32)),
}

VALUES = {
    "mixed": [0.0, -0.0, np.inf, -np.inf, 1.0, -1.0, 2.0, -3.0, np.nan, 1e-40, -1e-40],
    "subzero": [0.0, -0.0, 1e-40, -1e-40, 1e-41, -1e-41],
}


def _input(vals, shape, seed=0):
    return np.random.default_rng(seed).choice(np.array(vals, np.float32), size=shape).astype(np.float32)


@pytest.mark.parametrize("vname", sorted(VALUES))
@pytest.mark.parametrize("wname", sorted(WINDOWS))
def test_window_extrema_flush_subnormals(wname, vname):
    dev = _mps()
    is_max, dims, strides, pad, shape = WINDOWS[wname]
    x = _input(VALUES[vname], shape)
    pads = tuple(lax.padtype_to_pads(shape, dims, strides, pad))

    def f(v):
        init = jnp.asarray(-jnp.inf if is_max else jnp.inf, v.dtype)
        return lax.reduce_window(v, init, lax.max if is_max else lax.min, dims, strides, pad)

    got = np.asarray(jax.jit(f)(jax.device_put(x, dev)))
    ref = _ref(x, is_max, dims, strides, pads)
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("vname", sorted(VALUES))
@pytest.mark.parametrize(("is_max", "axis", "reverse"), [(True, 0, False), (False, 1, False), (True, 1, True)])
def test_cumulative_extrema_flush_subnormals(is_max, axis, reverse, vname):
    dev = _mps()
    x = _input(VALUES[vname], (32, 32), seed=1)
    op = lax.cummax if is_max else lax.cummin
    got = np.asarray(jax.jit(lambda v: op(v, axis=axis, reverse=reverse))(jax.device_put(x, dev)))
    xr = np.flip(x, axis) if reverse else x
    n = x.shape[axis]
    dims = tuple(n if a == axis else 1 for a in range(2))
    pads = tuple((n - 1, 0) if a == axis else (0, 0) for a in range(2))
    ref = _ref(np.ascontiguousarray(xr), is_max, dims, (1, 1), pads)
    ref = np.flip(ref, axis) if reverse else ref
    np.testing.assert_array_equal(got.view(np.uint32), np.ascontiguousarray(ref).view(np.uint32))


# Single-position windows, compared to XLA:CPU directly: (dims, strides, padding).
# The all-ones window keeps subnormals on XLA:CPU; the others pin down where that stops.
SINGLE = {
    "identity": ((1, 1), (1, 1), ((0, 0), (0, 0))),
    "stride2": ((1, 1), (2, 1), ((0, 0), (0, 0))),
    "padded": ((1, 1), (1, 1), ((1, 0), (0, 1))),
}


@pytest.mark.parametrize("init", ["ident", 0.0, -0.0, 1e-40, -1e-40])
@pytest.mark.parametrize("sname", sorted(SINGLE))
@pytest.mark.parametrize("is_max", [True, False])
def test_single_position_window_matches_cpu(is_max, sname, init):
    dev = _mps()
    dims, strides, pads = SINGLE[sname]
    x = _input(VALUES["mixed"], (16, 16), seed=2)
    init_val = (-np.inf if is_max else np.inf) if init == "ident" else init

    def f(v):
        return lax.reduce_window(v, jnp.asarray(init_val, v.dtype), lax.max if is_max else lax.min, dims, strides, pads)

    got = np.asarray(jax.jit(f)(jax.device_put(x, dev)))
    want = np.asarray(jax.jit(f)(jax.device_put(x, jax.devices("cpu")[0])))
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
