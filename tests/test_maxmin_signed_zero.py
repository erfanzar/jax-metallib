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
"""Elementwise max/min/clamp order -0 < +0 like XLA:CPU, in either operand order.

XLA:CPU returns max(-0, +0) == +0 and min(-0, +0) == -0 regardless of operand order.
MPSGraph maximum/minimum return the secondary operand on a tie and canonicalise the
operand order, so jnp.maximum(-0., +0.) and clip(+0., -0., 1.) came back with the wrong
sign. JAX_METALLIB_DISABLE_MAXMIN_ZERO_TIE=1 restores the old lowering.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

_UINT = {2: np.uint16, 4: np.uint32}
_DTYPES = [jnp.float32, jnp.float16, jnp.bfloat16]


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _arr(seed, n, dtype, pool=(-0.0, 0.0, -0.0, 0.0, -1.0, 2.0, -0.5)):
    rng = np.random.default_rng(seed)
    return np.asarray(jnp.asarray(np.array(pool, np.float32)[rng.integers(0, len(pool), n)], dtype))


def _compare(fn, args):
    cpu, mps = _devices()
    jf = jax.jit(fn)
    want = np.asarray(jf(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jf(*[jax.device_put(a, mps) for a in args]))
    assert got.shape == want.shape and got.dtype == want.dtype
    u = _UINT[want.dtype.itemsize]
    wb, gb = want.reshape(-1).view(u), got.reshape(-1).view(u)
    bad = np.nonzero(wb != gb)[0]
    assert bad.size == 0, f"{bad.size} bit mismatches, first at {bad[0]}: cpu={wb[bad[0]]:#x} mps={gb[bad[0]]:#x}"
    return want


def _tie_pairs(dtype):
    a = np.asarray(jnp.asarray([-0.0, -0.0, 0.0, 0.0, -1.0, 1.0], dtype))
    b = np.asarray(jnp.asarray([-0.0, 0.0, -0.0, 0.0, -0.0, 0.0], dtype))
    return a, b


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("swap", [False, True])
def test_two_tensor_ties(op, swap, dtype):
    a, b = _tie_pairs(dtype)
    if swap:
        a, b = b, a
    fn = lax.max if op == "max" else lax.min
    want = _compare(fn, [a, b]).astype(np.float32)
    expect_sign = [True, False, False, False] if op == "max" else [True, True, True, False]
    assert list(np.signbit(want[:4])) == expect_sign


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("const", [0.0, -0.0])
@pytest.mark.parametrize("const_first", [False, True])
def test_zero_constant_operand(op, const, const_first, dtype):
    fn = jnp.maximum if op == "max" else jnp.minimum
    c = jnp.dtype(dtype).type(const)

    def f(x):
        return fn(c, x) if const_first else fn(x, c)

    _compare(f, [_arr(0, 64, dtype)])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_relu_and_nonzero_constant(dtype):
    x = _arr(1, 64, dtype)
    _compare(lambda v: jnp.maximum(v, 0.0), [x])
    _compare(lambda v: jnp.maximum(v, -1.0), [x])
    _compare(lambda v: jnp.minimum(v, 2.0), [x])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("lo", [-0.0, 0.0])
def test_clip_zero_bound(lo, dtype):
    d = jnp.dtype(dtype).type
    _compare(lambda v: jnp.clip(v, d(lo), d(1.0)), [_arr(2, 64, dtype)])
    _compare(lambda v: lax.clamp(d(-1.0), v, d(lo)), [_arr(3, 64, dtype)])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_clamp_tensor_bounds(dtype):
    _compare(lambda lo, x, hi: lax.clamp(lo, x, hi), [_arr(4, 64, dtype), _arr(5, 64, dtype), _arr(6, 64, dtype)])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_random_tensors_bitwise(dtype):
    a, b = _arr(7, 4096, dtype), _arr(8, 4096, dtype)
    _compare(lax.max, [a, b])
    _compare(lax.min, [a, b])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_chain(dtype):
    # max/min inside an elementwise chain (fusion candidates), with -0 reaching the tie.
    def f(a, b):
        return jnp.minimum(jnp.maximum(a * 0.5, b), -b) * 2.0

    _compare(f, [_arr(9, 1024, dtype), _arr(10, 1024, dtype)])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_nan_still_propagates(dtype):
    a = np.asarray(jnp.asarray([np.nan, 0.0, -0.0, np.nan], dtype))
    b = np.asarray(jnp.asarray([0.0, np.nan, np.nan, -0.0], dtype))
    _cpu, mps = _devices()
    for fn in (lax.max, lax.min):
        got = np.asarray(jax.jit(fn)(jax.device_put(a, mps), jax.device_put(b, mps))).astype(np.float32)
        assert np.all(np.isnan(got))
