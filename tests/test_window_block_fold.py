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
"""Float reduce_window max/min block fold matches XLA:CPU bitwise.

The float order-key window fold replaces strided slices on an undilated axis
whose window is a multiple of its stride (w = m*s, s > 1) with a crop, a
reshape to [o + m - 1, s] and one max/min reduction over the s axis, followed
by a stride-1 fold of m (JAX_METALLIB_DISABLE_WINDOW_BLOCK_FOLD=1 keeps the
strided slices). These cases pin the crop (odd extents, padding that reaches
into a block), k = s / 2s / 3s, mixed blocked and stride-1 axes, dilated axes
(never blocked), remainder axes k = m*s + r (JAX_METALLIB_DISABLE_WINDOW_REMAINDER_FOLD=1
keeps their strided slices), window-1 axes with a stride, rank 1, a
non-identity init and an elementwise producer, with -0/+0, +-inf, NaN and
subnormals.

Shapes stay <= 33 per axis: from 64x64 on, XLA:CPU 2x2 stride-2 pooling drops
NaN and misorders zeros (upstream).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32}
DTYPES = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}

# name: (input shape, window, strides, padding, window_dilation)
CASES = {
    "2x2s2_valid_even": ((32, 32), (2, 2), (2, 2), "VALID", None),
    "2x2s2_valid_odd": ((31, 33), (2, 2), (2, 2), "VALID", None),
    "2x2s2_same_odd": ((31, 33), (2, 2), (2, 2), "SAME", None),
    "2x2s2_explicit_pad": ((30, 29), (2, 2), (2, 2), [(1, 0), (0, 3)], None),
    "1x4s4_crop": ((16, 30), (1, 4), (1, 4), "VALID", None),
    "4x1s4_same": ((30, 16), (4, 1), (4, 1), "SAME", None),
    "1x4s2_k2s": ((16, 31), (1, 4), (1, 2), "VALID", None),
    "4x1s2_k2s_same": ((31, 16), (4, 1), (2, 1), "SAME", None),
    "1x6s2_k3s": ((8, 33), (1, 6), (1, 2), "VALID", None),
    "2x2s21_mixed": ((31, 32), (2, 2), (2, 1), "VALID", None),
    "2x3s2x1_mixed_same": ((29, 31), (2, 3), (2, 1), "SAME", None),
    "4x4s2_two_blocked": ((32, 30), (4, 4), (2, 2), "SAME", None),
    "8x8s8_half": ((32, 32), (8, 8), (8, 8), "VALID", None),
    "1x3s2_not_multiple": ((16, 31), (1, 3), (1, 2), "VALID", None),
    "1x2s2_dilated": ((16, 32), (1, 2), (1, 2), "VALID", (1, 2)),
    "rank1_4s2": ((33,), (4,), (2,), "SAME", None),
    "rank1_2s2": ((31,), (2,), (2,), "VALID", None),
    # A window-1 axis with stride > 1 next to a fully blocked axis: the blocks
    # alone are not the output, the unit axis still has to be subsampled.
    "1x2s2x2_unit_axis": ((31, 33), (1, 2), (2, 2), "VALID", None),
    "2x1s2x2_unit_axis_same": ((31, 33), (2, 1), (2, 2), "SAME", None),
    "1x4s3x2_unit_axis": ((31, 33), (1, 4), (3, 2), "VALID", None),
    "1x4s2x4_unit_axis_pad": ((30, 29), (1, 4), (2, 4), [(1, 0), (0, 3)], None),
    # Remainder axes (w = m*s + r): one [o + m, s] reshape, full-block and
    # first-r-column reductions. Covers tail pad (nb*s > n), crop (nb*s < n),
    # r > 1, m = 2 and a remainder axis next to a blocked one.
    "3x3s2_rem_same": ((31, 33), (3, 3), (2, 2), "SAME", None),
    "1x5s3_rem_crop": ((16, 34), (1, 5), (1, 3), "VALID", None),
    "1x5s3_r2_pad": ((16, 31), (1, 5), (1, 3), [(0, 0), (2, 1)], None),
    "3x1s2_rem_pad": ((30, 29), (3, 1), (2, 1), [(1, 0), (0, 0)], None),
    "2x3s2_block_and_rem": ((31, 33), (2, 3), (2, 2), "VALID", None),
    "1x7s3_m2_rem": ((8, 33), (1, 7), (1, 3), "VALID", None),
    "rank1_5s2_same": ((33,), (5,), (2,), "SAME", None),
    # Rule R* (round 20g): unpadded, undilated f32 strided windows of 5..64
    # positions with one windowed axis, or only pure-block windowed axes, take
    # the order key instead of pooling (JAX_METALLIB_DISABLE_F32_STRIDED_KEY_WIDEN=1
    # restores the cap of 4). f16/bf16 already took these.
    "rs_1x9s2_rem": ((8, 33), (1, 9), (1, 2), "VALID", None),
    "rs_10x1s3_rem": ((33, 8), (10, 1), (3, 1), "VALID", None),
    "rs_1x6s5_rem": ((8, 33), (1, 6), (1, 5), "VALID", None),
    "rs_16x1s8_block": ((33, 8), (16, 1), (8, 1), "VALID", None),
    "rs_2x6s2_blocks": ((31, 33), (2, 6), (2, 2), "VALID", None),
    "rs_4x8s4_blocks": ((32, 33), (4, 8), (4, 4), "VALID", None),
    "rs_3x6s3_blocks": ((31, 33), (3, 6), (3, 3), "VALID", None),
    "rs_rank1_9s4": ((33,), (9,), (4,), "VALID", None),
    # Out of R*: two windowed axes with a remainder, and a stride-1 windowed axis
    # next to a strided window-1 axis (1.75-2.04x slower on the key); both stay
    # on pooling for f32 but take the key for f16/bf16.
    "rx_4x5s2_excluded": ((31, 33), (4, 5), (2, 2), "VALID", None),
    "rx_1x8s2x1_unit_strided": ((31, 33), (1, 8), (2, 1), "VALID", None),
    "rx_8x1s1x2_unit_strided": ((33, 31), (8, 1), (1, 2), "VALID", None),
    # w < s (remainder fold with m = 0): excluded from R* for f32.
    "rx_1x5s8_w_lt_s": ((8, 33), (1, 5), (1, 8), "VALID", None),
    "rx_6x1s8_w_lt_s": ((33, 8), (6, 1), (8, 1), "VALID", None),
}


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
        f"{bad.size} mismatches, first at {bad[:4]}: cpu={want.reshape(-1)[bad[:4]]} mps={got.reshape(-1)[bad[:4]]}"
    )
    return want


def _specials(dt, shape, seed):
    vals = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 2.0, -3.0, 0.5]
    if dt != jnp.bfloat16:
        vals += [1e-40, -1e-40] if dt == jnp.float32 else [1e-6, -1e-6]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    mask = rng.random(shape) < 0.5
    x[mask] = rng.choice(np.array(vals, np.float32), size=int(mask.sum()))
    return x.astype(dt)


def _fn(kind, init, win, strides, pad, wdil, relu):
    op = lax.max if kind == "max" else lax.min

    def fn(x):
        if relu:
            x = jnp.maximum(x, jnp.zeros((), x.dtype))
        return lax.reduce_window(x, jnp.asarray(init, x.dtype), op, win, strides, pad, window_dilation=wdil)

    return fn


@pytest.mark.parametrize("producer", ["none", "relu"])
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("case", list(CASES))
def test_block_fold_specials(case, dtname, kind, producer):
    dt = DTYPES[dtname]
    shape, win, strides, pad, wdil = CASES[case]
    init = -np.inf if kind == "max" else np.inf
    x = _specials(dt, shape, seed=len(case))
    _check(_fn(kind, init, win, strides, pad, wdil, producer == "relu"), x)


@pytest.mark.parametrize("init", [0.5, -0.0, 0.0])
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("case", ["2x2s2_same_odd", "1x4s2_k2s", "rank1_4s2"])
def test_block_fold_non_identity_init(case, dtname, kind, init):
    # The init is folded after the block fold and is also the pad value.
    dt = DTYPES[dtname]
    shape, win, strides, pad, wdil = CASES[case]
    x = _specials(dt, shape, seed=7)
    _check(_fn(kind, init, win, strides, pad, wdil, False), x)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("dtname", list(DTYPES))
def test_block_fold_zero_ties(dtname, kind):
    # Every 2x2 block holds both zeros: a fold that keeps the first zero met
    # returns the losing sign on half the blocks.
    dt = DTYPES[dtname]
    r = np.arange(31)[:, None]
    c = np.arange(33)[None, :]
    x = np.where((r + c) % 2 == 0, -0.0, 0.0).astype(np.float32).astype(dt)
    init = -np.inf if kind == "max" else np.inf
    want = _check(_fn(kind, init, (2, 2), (2, 2), "SAME", None, False), x)
    # SAME pads one row and one column high, so the corner window holds only
    # x[30, 32] (-0); every other window holds both zeros.
    w = np.asarray(want, np.float32)
    assert w.shape == (16, 17)
    w = np.delete(w.reshape(-1), w.size - 1)
    assert (np.signbit(w) == (kind == "min")).all()
