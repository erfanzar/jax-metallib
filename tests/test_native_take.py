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

"""jnp.take(table, i, axis=0) runs as one native wrap/bounds/copy-or-fill step.

FuseTakePatterns rewrites JAX's take lowering (negative-index wrap, bounds
mask, row gather, fill select) into a tagged gather. Results must be
bit-identical to XLA:CPU, including NaN payloads, signed zeros, negative and
out-of-range indices and custom fill values. Forms the pass does not match
(bf16 or integer tables, clip mode, other axes, 2-D indices) must stay correct.
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


def _assert_bitwise(fn, *args):
    out = jax.tree.leaves(_run(fn, *args, device=MPS))
    ref = jax.tree.leaves(_run(fn, *args, device=CPU))
    assert len(out) == len(ref)
    for o, r in zip(out, ref, strict=True):
        assert o.shape == r.shape and o.dtype == r.dtype
        np.testing.assert_array_equal(_bits(o), _bits(r))


def _table(rows, cols, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return rng.integers(-1000, 1000, size=(rows, cols)).astype(dtype)
    t = rng.standard_normal((rows, cols)).astype(np.float32)
    t.flat[::11] = -0.0
    t.flat[2] = np.nan
    t = t.astype(dtype)
    if np.dtype(dtype) == np.float32:
        t.view(np.uint32).flat[5] = 0xFFC00123  # negative NaN with payload
    elif np.dtype(dtype) == np.float16:
        t.view(np.uint16).flat[5] = 0xFE01
    return t


def _indices(rows, n, seed=1):
    rng = np.random.default_rng(seed)
    i = rng.integers(-2 * rows, 2 * rows, size=(n,)).astype(np.int32)
    edges = [0, -1, rows - 1, rows, -rows, -rows - 1, np.iinfo(np.int32).min, np.iinfo(np.int32).max]
    i[: len(edges)] = edges
    return i


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
@pytest.mark.parametrize(("rows", "cols", "n"), [(4096, 512, 1024), (37, 5, 64), (1000, 3, 9), (1, 7, 12)])
def test_take_rows_bitwise(dtype, rows, cols, n):
    t = _table(rows, cols, dtype)
    i = _indices(rows, n)
    _assert_bitwise(lambda e, k: jnp.take(e, k, axis=0), t, i)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("fill", [7.0, -0.0, np.inf])
def test_take_custom_fill_value(dtype, fill):
    t = _table(64, 16, dtype, 2)
    i = _indices(64, 40, 3)
    _assert_bitwise(lambda e, k: jnp.take(e, k, axis=0, mode="fill", fill_value=fill), t, i)


@pytest.mark.parametrize(
    "dtype", [jnp.bfloat16, np.int32, np.int16, np.float64], ids=["bf16", "i32", "i16", "f64-as-f32"]
)
def test_take_unfused_dtypes(dtype):
    t = _table(50, 8, np.float32 if dtype == np.float64 else dtype, 4)
    i = _indices(50, 30, 5)
    _assert_bitwise(lambda e, k: jnp.take(e, k, axis=0), t, i)


def test_take_clip_mode_and_other_axis():
    t = _table(40, 24, np.float32, 6)
    i = _indices(40, 16, 7)
    j = _indices(24, 10, 8)

    def fn(e, k, m):
        return jnp.take(e, k, axis=0, mode="clip"), jnp.take(e, m, axis=1), e.at[jnp.abs(k) % 40].get()

    _assert_bitwise(fn, t, i, j)


def test_take_2d_indices():
    t = _table(100, 32, np.float32, 9)
    i = _indices(100, 60, 10).reshape(6, 10)
    _assert_bitwise(lambda e, k: jnp.take(e, k, axis=0), t, i)


def test_take_feeds_more_ops_and_is_returned():
    t = _table(512, 64, np.float32, 11)
    i = _indices(512, 128, 12)

    def fn(e, k):
        rows = jnp.take(e, k, axis=0)
        return rows, jnp.nan_to_num(rows) * 2.0 + 1.0, jnp.take(e, k[::-1], axis=0)

    _assert_bitwise(fn, t, i)


def test_two_takes_from_same_table():
    t = _table(256, 16, np.float16, 13)
    a = _indices(256, 40, 14)
    b = _indices(256, 40, 15)
    _assert_bitwise(lambda e, x, y: (jnp.take(e, x, axis=0), jnp.take(e, y, axis=0)), t, a, b)


def test_repeated_calls_new_indices():
    f = jax.jit(lambda e, k: jnp.take(e, k, axis=0))
    t = _table(300, 20, np.float32, 16)
    for seed in range(3):
        i = _indices(300, 50, 20 + seed)
        out = f(jax.device_put(t, MPS), jax.device_put(i, MPS))
        ref = f(jax.device_put(t, CPU), jax.device_put(i, CPU))
        np.testing.assert_array_equal(_bits(out), _bits(ref))


def test_grad_through_embedding():
    rng = np.random.default_rng(17)
    t = rng.standard_normal((128, 32)).astype(np.float32)
    i = rng.integers(0, 128, size=(64,)).astype(np.int32)

    def loss(e, k):
        return jnp.sum(jnp.tanh(jnp.take(e, k, axis=0)) ** 2)

    out = np.asarray(_run(jax.grad(loss), t, i, device=MPS))
    ref = np.asarray(_run(jax.grad(loss), t, i, device=CPU))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
