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

"""Short counted while loops are unrolled when the module is parsed.

The body is cloned per trip with a constant counter, scalar integer index
math is folded, and constant-start dynamic_slice becomes slice. Results must
match XLA:CPU, including JAX's negative-index wrap and XLA start clamping, and
loops the pass must not touch (too many trips, buffer writes, nested control
flow) must still be correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]


def _run(fn, *args, device):
    return jax.jit(fn)(*(jax.device_put(x, device) for x in args))


def _bits(x):
    x = np.asarray(x)
    return x.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[x.dtype.itemsize])


def _close(out, ref, dtype):
    out = np.asarray(out, np.float64)
    ref = np.asarray(ref, np.float64)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    tol = 1e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("layers", [1, 2, 6, 16])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16], ids=["f32", "f16", "bf16"])
def test_scan_over_stacked_weights(layers, dtype):
    rng = np.random.default_rng(layers)
    x = rng.standard_normal((8, 32)).astype(dtype)
    w = (rng.standard_normal((layers, 32, 32)) * 0.2).astype(dtype)
    b = rng.standard_normal((layers, 32)).astype(dtype)

    def fn(x, w, b):
        def body(c, i):
            return jnp.tanh(c @ w[i] + b[i]), None

        return jax.lax.scan(body, x, jnp.arange(layers))[0]

    _close(_run(fn, x, w, b, device=MPS), _run(fn, x, w, b, device=CPU), dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.uint8], ids=["f32", "f16", "i32", "u8"])
def test_scan_slices_are_bitwise(dtype):
    # Pure data movement: every slice must be bit-identical to CPU.
    rng = np.random.default_rng(1)
    if np.issubdtype(dtype, np.floating):
        w = rng.standard_normal((5, 3, 7)).astype(dtype)
        w.flat[::9] = -0.0
        w.flat[4] = np.nan
    else:
        w = rng.integers(0, 120, size=(5, 3, 7)).astype(dtype)

    def fn(w):
        def body(c, i):
            return c, w[i]

        return jax.lax.scan(body, jnp.zeros((), w.dtype), jnp.arange(5))[1]

    out, ref = _run(fn, w, device=MPS), _run(fn, w, device=CPU)
    np.testing.assert_array_equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("index", [-1, -6, 0, 3])
def test_negative_index_wrap(index):
    # JAX wraps negative indices (i < 0 -> i + n); the folded select must too.
    w = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)

    def fn(w):
        def body(c, i):
            return c + w[i + index], None

        return jax.lax.scan(body, jnp.zeros((4,), jnp.float32), jnp.arange(3))[0]

    np.testing.assert_array_equal(np.asarray(_run(fn, w, device=MPS)), np.asarray(_run(fn, w, device=CPU)))


@pytest.mark.parametrize("offset", [-9, 4, 7, 100])
def test_dynamic_slice_start_clamped(offset):
    # lax.dynamic_slice clamps out-of-range starts to [0, dim - size].
    w = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)

    def fn(w):
        def body(c, i):
            return c + jax.lax.dynamic_slice_in_dim(w, i + offset, 2, axis=0).sum(0), None

        return jax.lax.scan(body, jnp.zeros((3,), jnp.float32), jnp.arange(4))[0]

    np.testing.assert_array_equal(np.asarray(_run(fn, w, device=MPS)), np.asarray(_run(fn, w, device=CPU)))


@pytest.mark.parametrize(("lower", "upper", "step"), [(2, 9, 1), (0, 10, 3), (-4, 4, 2), (5, 6, 1)])
def test_fori_loop_bounds_and_step(lower, upper, step):
    w = np.arange(20 * 2, dtype=np.float32).reshape(20, 2)

    def fn(w):
        def body(k, carry):
            i, acc = carry
            return i + step, acc * 0.5 + w[i + 10]

        n = (upper - lower + step - 1) // step
        return jax.lax.fori_loop(0, n, body, (jnp.int32(lower), jnp.zeros((2,), jnp.float32)))[1]

    np.testing.assert_array_equal(np.asarray(_run(fn, w, device=MPS)), np.asarray(_run(fn, w, device=CPU)))


def test_while_with_explicit_counter_and_step():
    # Hand-written counted while: counter starts at 3, step 2, bound 13.
    w = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)

    def fn(w):
        def cond(s):
            return s[0] < 13

        def body(s):
            i, acc = s
            return i + 2, acc + w[i] * jnp.float32(i)

        return jax.lax.while_loop(cond, body, (jnp.int32(3), jnp.zeros((3,), jnp.float32)))[1]

    np.testing.assert_array_equal(np.asarray(_run(fn, w, device=MPS)), np.asarray(_run(fn, w, device=CPU)))


def test_int8_counter_near_limit():
    # 115, 119, 123, then 127 ends the loop without wrapping. The pass's guard
    # (upper - 1 + step = 130 > int8 max) must reject it conservatively, and
    # the result must still match. (Starting at 118 would wrap 130 -> -126,
    # which is < 127, and the loop would never end on any backend.)
    w = np.arange(256 * 2, dtype=np.float32).reshape(256, 2)

    def fn(w):
        def cond(s):
            return s[0] < jnp.int8(127)

        def body(s):
            i, n, acc = s
            return i + jnp.int8(4), n + 1, acc + w[i.astype(jnp.int32)]

        return jax.lax.while_loop(cond, body, (jnp.int8(115), jnp.int32(0), jnp.zeros((2,), jnp.float32)))[1:]

    out = _run(fn, w, device=MPS)
    ref = _run(fn, w, device=CPU)
    for o, r in zip(out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(o), np.asarray(r))


def test_long_loop_stays_a_loop():
    w = np.random.default_rng(3).standard_normal((40, 8)).astype(np.float32)

    def fn(w):
        def body(c, i):
            return jnp.sin(c) + w[i], None

        return jax.lax.scan(body, jnp.zeros((8,), jnp.float32), jnp.arange(40))[0]

    _close(_run(fn, w, device=MPS), _run(fn, w, device=CPU), np.float32)


def test_stacked_outputs_and_nested_loop():
    # Body with a buffer write (stacked ys) and a nested fori_loop.
    w = np.random.default_rng(4).standard_normal((4, 6)).astype(np.float32)

    def fn(w):
        def body(c, i):
            inner = jax.lax.fori_loop(0, 3, lambda k, v: v * 0.5 + w[i], c)
            return inner, inner.sum()

        return jax.lax.scan(body, jnp.ones((6,), jnp.float32), jnp.arange(4))

    out = _run(fn, w, device=MPS)
    ref = _run(fn, w, device=CPU)
    for o, r in zip(out, ref, strict=True):
        _close(o, r, np.float32)


def test_grad_through_scan():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 16)).astype(np.float32)
    w = (rng.standard_normal((3, 16, 16)) * 0.3).astype(np.float32)

    def loss(x, w):
        def body(c, i):
            return jnp.tanh(c @ w[i]), None

        return jnp.sum(jax.lax.scan(body, x, jnp.arange(3))[0] ** 2)

    out = _run(jax.grad(loss, argnums=(0, 1)), x, w, device=MPS)
    ref = _run(jax.grad(loss, argnums=(0, 1)), x, w, device=CPU)
    for o, r in zip(out, ref, strict=True):
        _close(o, r, np.float32)
