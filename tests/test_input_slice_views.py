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

"""Static contiguous slices of entry arguments are fed as MPSNDArray views.

A slice of a jit argument covering one contiguous row-major range is bound to
the graph as a view of the argument's buffer at a byte offset. Results must be
bit-identical to XLA:CPU for pure data movement, and slices that do not
qualify (strided, small, unaligned, used outside the segment) must still be
correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]

DTYPES = [np.float32, np.float16, jnp.bfloat16, np.int32]
IDS = ["f32", "f16", "bf16", "i32"]


def _run(fn, *args, device):
    return jax.jit(fn)(*(jax.device_put(x, device) for x in args))


def _bits(x):
    x = np.asarray(x)
    return x.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[x.dtype.itemsize])


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == np.int32:
        return rng.integers(-1000, 1000, size=shape).astype(np.int32)
    a = rng.standard_normal(shape).astype(np.float32)
    a.flat[::13] = -0.0
    a.flat[7] = np.nan
    return a.astype(dtype)


def _assert_bitwise(fn, *args):
    out = _run(fn, *args, device=MPS)
    ref = _run(fn, *args, device=CPU)
    out, ref = jax.tree.leaves(out), jax.tree.leaves(ref)
    assert len(out) == len(ref)
    for o, r in zip(out, ref, strict=True):
        assert o.shape == r.shape and o.dtype == r.dtype
        np.testing.assert_array_equal(_bits(o), _bits(r))


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_leading_axis_slices_with_reshape(dtype):
    # w[i] lowers to slice [i:i+1] + reshape; every layer becomes a view.
    w = _data((6, 128, 256), dtype)

    def fn(w):
        return [w[i] * 1 for i in range(6)]

    _assert_bitwise(fn, w)


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_inner_contiguous_range(dtype):
    # [2, 40:200, :] of [4, 256, 128]: one contiguous range at a nonzero offset.
    w = _data((4, 256, 128), dtype, 1)

    def fn(w):
        return w[2, 40:200, :] + w[3, 0:160, :]

    _assert_bitwise(fn, w)


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_slice_without_reshape(dtype):
    w = _data((8, 64, 512), dtype, 2)

    def fn(w):
        return jax.lax.slice(w, (3, 0, 0), (5, 64, 512)) * 1

    _assert_bitwise(fn, w)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
def test_non_qualifying_slices(dtype):
    # Strided, column-range (non-contiguous), tiny, and unaligned slices stay
    # in-graph; mixed with a viewed slice of the same argument in one graph.
    w = _data((6, 256, 256), dtype, 3)

    def fn(w):
        viewed = w[1]
        strided = w[0, ::2, :]
        columns = w[2, :, 10:200]
        tiny = w[3, 0:2, :]
        unaligned = w.reshape(-1)[1:70001]
        return viewed * 1, strided * 1, columns * 1, tiny * 1, unaligned * 1

    _assert_bitwise(fn, w)


def test_slice_result_returned_directly():
    # A slice that is itself an output is not a view (it leaves the segment).
    w = _data((4, 128, 256), np.float32, 4)

    def fn(w):
        return w[1], w[2] * 2

    _assert_bitwise(fn, w)


def test_same_argument_full_and_view():
    # The argument is also used whole in the same graph.
    w = _data((4, 128, 256), np.float32, 5)

    def fn(w):
        return w[2] + w.sum(axis=0), w * 3

    out = _run(fn, w, device=MPS)
    ref = _run(fn, w, device=CPU)
    for o, r in zip(out, ref, strict=True):
        np.testing.assert_allclose(np.asarray(o), np.asarray(r), rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16], ids=["f32", "f16", "bf16"])
def test_scan_over_stacked_weights_matmul(dtype):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((32, 128)).astype(dtype)
    w = (rng.standard_normal((5, 128, 128)) * 0.1).astype(dtype)

    def fn(x, w):
        def body(c, i):
            return jnp.tanh(c @ w[i]), None

        return jax.lax.scan(body, x, jnp.arange(5))[0]

    out = np.asarray(_run(fn, x, w, device=MPS), np.float64)
    ref = np.asarray(_run(fn, x, w, device=CPU), np.float64)
    tol = 1e-5 if dtype == np.float32 else 3e-2
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol)


def test_repeated_calls_and_new_buffers():
    # Views are rebuilt per call; a new argument buffer must be picked up.
    f = jax.jit(lambda w: w[1] * 2 + w[3])
    for seed in range(3):
        w = _data((4, 128, 256), np.float32, 10 + seed)
        out = f(jax.device_put(w, MPS))
        ref = jax.jit(lambda w: w[1] * 2 + w[3])(jax.device_put(w, CPU))
        np.testing.assert_array_equal(_bits(out), _bits(ref))


def test_grad_through_stacked_weights():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 128)).astype(np.float32)
    w = (rng.standard_normal((3, 128, 128)) * 0.2).astype(np.float32)

    def loss(x, w):
        h = x
        for i in range(3):
            h = jnp.tanh(h @ w[i])
        return jnp.sum(h**2)

    out = _run(jax.grad(loss, argnums=(0, 1)), x, w, device=MPS)
    ref = _run(jax.grad(loss, argnums=(0, 1)), x, w, device=CPU)
    for o, r in zip(out, ref, strict=True):
        np.testing.assert_allclose(np.asarray(o), np.asarray(r), rtol=1e-4, atol=1e-5)
