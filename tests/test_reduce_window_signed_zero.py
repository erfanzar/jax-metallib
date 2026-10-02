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
"""reduce_window max/min and cummax/cummin match XLA:CPU bitwise on -0/+0 ties.

XLA:CPU orders -0 < +0 in max/min. MPSGraph pooling, cumulative max/min and
plain maximum/minimum keep one zero on a tie, so every window/scan lowering
(pooling, sliding fold, strided general fold, short scan fold, cumulative scan,
init fold) returned the losing zero on alternating rows
(JAX_METALLIB_DISABLE_WINDOW_ZERO_TIE=1 restores the old lowerings).

Also pinned: a +0 sum init is folded in (it is not an additive identity:
+0 + -0 == +0), bf16 min pooling at rank 3/4 (it was rejected), and the
NaN / inf / non-zero cases that share the pooling indicator and the cumulative
state scan with the zero fix-up.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32}
DTYPES = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _bits(a):
    a = np.asarray(a)
    return a.reshape(-1).view(UINT[a.dtype.itemsize])


def _alternating(nrows, n):
    # Row r starts with -0 when r is even, +0 when odd: every window and prefix
    # sees both (-0, +0) and (+0, -0).
    r = np.arange(nrows)[:, None]
    c = np.arange(n)[None, :]
    return np.where((r + c) % 2 == 0, -0.0, 0.0).astype(np.float32)


def _mixed(n):
    # Rows that mix zeros with values that must outrank the zero fix-up.
    rows = [
        [0.0, 1.0, -0.0, 0.0],  # finite non-zero beats the zero bit (max)
        [-0.0, -1.0, 0.0, -0.0],  # finite non-zero beats the zero bit (min)
        [-np.inf, -0.0, 0.0, -0.0],  # losing inf then zeros
        [np.inf, 0.0, -0.0, 0.0],  # losing inf (min) then zeros
        [-0.0, np.nan, 0.0, -0.0],  # NaN wins from its position on
        [0.0, -0.0, np.inf, -np.inf],  # winning inf after a tie
        [-0.0, -0.0, -0.0, -0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    base = np.array(rows, np.float32)
    reps = -(-n // base.shape[1])
    return np.tile(base, (1, reps))[:, :n]


def _check(fn, x):
    cpu, mps = _devices()
    want = np.asarray(jax.jit(fn)(jax.device_put(x, cpu)))
    got = np.asarray(jax.jit(fn)(jax.device_put(x, mps)))
    assert want.shape == got.shape and want.dtype == got.dtype
    assert np.array_equal(_bits(want), _bits(got)), f"cpu={want.reshape(-1)} mps={got.reshape(-1)}"
    return want


def _assert_tie_decided(want):
    # Vacuity guard: the reference holds both zeros, so a wrong tie rule fails.
    w = np.asarray(want, np.float32).reshape(-1)
    zeros = w[w == 0]
    assert np.signbit(zeros).any() and (~np.signbit(zeros)).any()


def _rw(init, op, win, strides, pad):
    return lambda x: lax.reduce_window(x, init, op, win, strides, pad)


def _window_cases(dt):
    d = jnp.dtype(dt)
    ninf, pinf = d.type(-np.inf), d.type(np.inf)
    x2 = _alternating(4, 6)
    return (
        {
            "w2_valid": ((1, 2), (1, 1), "VALID", x2),
            "h2_valid": ((2, 1), (1, 1), "VALID", x2),
            "w2_padlow": ((1, 2), (1, 1), [(0, 0), (1, 0)], x2),
            "r1_same3": ((3,), (1,), "SAME", x2[0]),
            "w2_s2": ((1, 2), (1, 2), "VALID", x2),
            "w3_s2_r1": ((3,), (2,), "VALID", np.concatenate([x2[0], x2[1]])),
            "nwc": ((1, 2, 1), (1, 1, 1), "VALID", x2.reshape(1, 24, 1)),
            "nhwc": ((1, 2, 2, 1), (1, 1, 1, 1), "VALID", x2.reshape(1, 4, 6, 1)),
            "nhwc_s2": ((1, 2, 2, 1), (1, 2, 2, 1), "VALID", x2.reshape(1, 4, 6, 1)),
        },
        ninf,
        pinf,
    )


WINDOW_NAMES = list(_window_cases(jnp.float32)[0])


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("case", WINDOW_NAMES)
def test_window_zero_tie(dtname, kind, case):
    dt = DTYPES[dtname]
    cases, ninf, pinf = _window_cases(dt)
    win, strides, pad, x = cases[case]
    fn = _rw(ninf, lax.max, win, strides, pad) if kind == "max" else _rw(pinf, lax.min, win, strides, pad)
    want = _check(fn, x.astype(dt))
    w = np.asarray(want, np.float32).reshape(-1)
    # The winning zero must appear in the reference (+0 for max, -0 for min).
    zeros = w[w == 0]
    assert zeros.size and (np.signbit(zeros).any() if kind == "min" else (~np.signbit(zeros)).any())


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 16, 64])
def test_scan_zero_tie(dtname, kind, reverse, n):
    dt = DTYPES[dtname]
    scan = lax.cummax if kind == "max" else lax.cummin
    want = _check(lambda x: scan(x, 1, reverse=reverse), _alternating(4, n).astype(dt))
    _assert_tie_decided(want)


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("n", [4, 9, 64])
def test_scan_mixed_specials(dtname, kind, reverse, n):
    # NaN, +-inf and non-zero finite values share the state scan / fold with
    # the zero fix-up and must keep precedence over it.
    dt = DTYPES[dtname]
    scan = lax.cummax if kind == "max" else lax.cummin
    _check(lambda x: scan(x, 1, reverse=reverse), _mixed(n).astype(dt))


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("case", ["w2_valid", "w2_padlow", "w2_s2", "nwc"])
def test_window_mixed_specials(dtname, kind, case):
    dt = DTYPES[dtname]
    cases, ninf, pinf = _window_cases(dt)
    win, strides, pad, x = cases[case]
    # Every case here has 24 elements: 8 mixed rows of 3, reshaped.
    x = _mixed(x.size // 8).reshape(x.shape)
    fn = _rw(ninf, lax.max, win, strides, pad) if kind == "max" else _rw(pinf, lax.min, win, strides, pad)
    _check(fn, x.astype(dt))


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("init", [-0.0, 0.0])
def test_zero_init_ties(dtname, kind, init):
    # A +-0 init folds in with the tie rule (cummax-style prefix and sliding).
    dt = DTYPES[dtname]
    op = lax.max if kind == "max" else lax.min
    iv = jnp.dtype(dt).type(init)
    x = _alternating(4, 6).astype(dt)
    _check(_rw(iv, op, (1, 6), (1, 1), [(0, 0), (5, 0)]), x)
    _check(_rw(iv, op, (1, 2), (1, 1), "VALID"), x)
    _check(_rw(iv, op, (1, 1), (1, 1), "VALID"), x)


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("case", ["w2_valid", "h2_valid", "nhwc"])
def test_sum_positive_zero_init(dtname, case):
    # +0 init is not an additive identity: a window of [-0, -0] sums to +0.
    dt = DTYPES[dtname]
    win, strides, pad, _ = _window_cases(dt)[0][case]
    x = np.full((4, 6), -0.0, np.float32)
    if case == "nhwc":
        x = x.reshape(1, 4, 6, 1)
    want = _check(_rw(jnp.dtype(dt).type(0.0), lax.add, win, strides, pad), x.astype(dt))
    assert not np.signbit(np.asarray(want, np.float32)).any()


# Float order-key window fold (rank <= 2; f32 stride-1 <= 64 positions, f32
# strided <= 4, f16/bf16 <= 64; JAX_METALLIB_DISABLE_FLOAT_WINDOW_KEY=1 keeps
# the pooling / per-step lowerings; block-fold cases: test_window_block_fold.py). Shapes stay <= 32x32: from 64x64 on,
# XLA:CPU 2x2 stride-2 pooling drops NaN and misorders zeros (upstream).
KEY_GATE_CASES = {
    "slide1x3": ((1, 3), (1, 1), "SAME"),
    "slide5x1": ((5, 1), (1, 1), "SAME"),
    "slide1x16_in": ((1, 16), (1, 1), "SAME"),
    "pool5x5s1_out_f32": ((5, 5), (1, 1), "SAME"),
    "pool2x2s2_in": ((2, 2), (2, 2), "VALID"),
    "pool3x3s2_out_f32": ((3, 3), (2, 2), "SAME"),
    "pool8x8s8_half_in": ((8, 8), (8, 8), "VALID"),
    "dilated": ((1, 2), (1, 1), [(0, 0), (1, 1)]),
}


def _specials(dt, shape, subnormals, seed=0):
    vals = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 2.0, -3.0]
    if subnormals:
        vals += [1e-40, -1e-40] if dt == jnp.float32 else [1e-6, -1e-6]
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(vals, np.float32), size=shape).astype(dt)


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("case", list(KEY_GATE_CASES))
def test_key_gate_mixed_specials(dtname, kind, case):
    # +-0, +-inf, NaN and (f32/f16) subnormals on both sides of the gate.
    dt = DTYPES[dtname]
    win, strides, pad = KEY_GATE_CASES[case]
    d = jnp.dtype(dt)
    x = _specials(dt, (32, 32), subnormals=dt != jnp.bfloat16)
    fn = (
        _rw(d.type(-np.inf), lax.max, win, strides, pad)
        if kind == "max"
        else _rw(d.type(np.inf), lax.min, win, strides, pad)
    )
    _check(fn, x)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("case", ["slide1x3", "slide5x1", "pool2x2s2_in", "dilated"])
def test_key_f32_subnormal_flush(kind, case):
    # XLA:CPU flushes f32 subnormals to a signed zero in window max/min; the
    # pooling path kept them (+-1e-40 came back unchanged).
    win, strides, pad = KEY_GATE_CASES[case]
    rng = np.random.default_rng(1)
    x = rng.choice(np.array([1e-40, -1e-40, 0.0, -0.0], np.float32), size=(16, 16))
    op, init = (lax.max, np.float32(-np.inf)) if kind == "max" else (lax.min, np.float32(np.inf))
    want = _check(_rw(init, op, win, strides, pad), x)
    # Vacuity guard: the input holds subnormals and the reference flushed them.
    w = np.asarray(want, np.float32)
    assert (np.abs(x) > 0).any() and not ((np.abs(w) > 0) & (np.abs(w) < 1.2e-38)).any()


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("init", [5.0, -0.0, 0.0, "inf"])
def test_key_all_padding_windows(dtname, kind, init):
    # Windows that lie entirely in padding return init: the key pads with
    # key(-inf)/key(+inf), because INT_MIN/INT_MAX decode to NaN.
    dt = DTYPES[dtname]
    d = jnp.dtype(dt)
    if init == "inf":
        init = -np.inf if kind == "max" else np.inf
    op = lax.max if kind == "max" else lax.min
    x = _mixed(8).astype(dt)
    fn = _rw(d.type(init), op, (1, 2), (1, 1), [(0, 0), (3, 3)])
    want = _check(fn, x)
    # First and last two output columns see only padding.
    edge = np.asarray(want, np.float32)[:, [0, 1, -2, -1]]
    assert np.array_equal(edge.view(np.uint32), np.full_like(edge, np.float32(init)).view(np.uint32))


@pytest.mark.parametrize("dt", [np.int32, np.int8, np.uint16])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_integer_windows_unchanged(dt, kind):
    rng = np.random.default_rng(0)
    info = np.iinfo(dt)
    x = rng.integers(max(info.min, -50), min(info.max, 50), size=(4, 12)).astype(dt)
    op = lax.max if kind == "max" else lax.min
    init = info.min if kind == "max" else info.max
    scan = lax.cummax if kind == "max" else lax.cummin
    _check(lambda v: scan(v, 1), x)
    _check(_rw(dt(init), op, (1, 3), (1, 2), "VALID"), x)
