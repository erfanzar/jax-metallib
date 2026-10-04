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
"""select whose operands are broadcast_in_dim results (masks, fill values, permuted broadcasts) matches CPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _mps():
    try:
        return jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")


def _check(fn, *args):
    mps = _mps()
    cpu = jax.devices("cpu")[0]
    want = jax.jit(fn)(*[jax.device_put(a, cpu) for a in args])
    got = jax.jit(fn)(*[jax.device_put(a, mps) for a in args])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_take_fill_out_of_bounds():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((64, 16)).astype(np.float32)
    idx = np.array([0, 5, 63, 64, -1, -65, 100, 7], dtype=np.int32)
    _check(lambda e, i: jnp.take(e, i, axis=0, mode="fill"), emb, idx)


def test_row_mask_scalar_fill():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 24)).astype(np.float32)
    m = rng.random(32) > 0.5
    _check(lambda x, m: jnp.where(m[:, None], x, jnp.float32(-3.0)), x, m)


def test_column_mask_row_values():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 12, 20)).astype(np.float32)
    m = rng.random(20) > 0.5
    v = rng.standard_normal(12).astype(np.float32)
    _check(lambda x, m, v: jnp.where(m, x, v[:, None]), x, m, v)


def test_all_broadcast_operands_keep_shape():
    rng = np.random.default_rng(3)
    m = rng.random(6) > 0.5
    a = rng.standard_normal(6).astype(np.float32)
    b = rng.standard_normal(5).astype(np.float32)

    def f(m, a, b):
        return jnp.where(m[:, None], a[:, None], b[None, :])

    _check(f, m, a, b)


def test_transposed_broadcast_operand():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((6, 7)).astype(np.float32)
    y = rng.standard_normal((7, 6)).astype(np.float32)
    m = rng.random((6, 7)) > 0.5

    def f(x, y, m):
        yt = jax.lax.broadcast_in_dim(y, (6, 7), (1, 0))
        return jnp.where(m, x, yt)

    _check(f, x, y, m)


def test_int_and_bool_values():
    rng = np.random.default_rng(5)
    x = rng.integers(-100, 100, size=(16, 9), dtype=np.int32)
    m = rng.random(9) > 0.5
    _check(lambda x, m: jnp.where(m, x, jnp.int32(7)), x, m)
    b = rng.random((16, 9)) > 0.5
    _check(lambda b, m: jnp.where(m, b, False), b, m)


def test_grad_through_masked_select():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((10, 4)).astype(np.float32)
    m = rng.random(10) > 0.5
    _check(jax.grad(lambda x, m: jnp.sum(jnp.where(m[:, None], x * x, 0.0))), x, m)
