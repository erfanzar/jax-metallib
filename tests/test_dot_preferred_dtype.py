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
"""Contractions whose declared result dtype is wider than the operands match CPU.

bf16 x bf16 dot/dot_general/einsum with preferred_element_type=float32 stored the
bf16 matmul result under the f32 result type, so it was read back as garbage
(relative error ~1.0-1.5, no error raised). f16 and integer variants are kept as
controls.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _arr(rng, shape, dtype):
    if jnp.issubdtype(dtype, jnp.integer):
        lo = -5 if jnp.issubdtype(dtype, jnp.signedinteger) else 0
        return rng.integers(lo, 6, shape).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


_KINDS = {
    "dot": (((33, 40), (40, 17)), lambda acc: lambda a, b: lax.dot(a, b, preferred_element_type=acc)),
    "dot_big": (((256, 512), (512, 128)), lambda acc: lambda a, b: lax.dot(a, b, preferred_element_type=acc)),
    "batched": (
        ((3, 9, 20), (3, 20, 7)),
        lambda acc: lambda a, b: lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))), preferred_element_type=acc),
    ),
    "einsum": (
        ((4, 6, 8), (8, 5)),
        lambda acc: lambda a, b: jnp.einsum("bij,jk->bik", a, b, preferred_element_type=acc),
    ),
    "matvec": (((31, 40), (40,)), lambda acc: lambda a, b: lax.dot(a, b, preferred_element_type=acc)),
    "vecvec": (((40,), (40,)), lambda acc: lambda a, b: lax.dot(a, b, preferred_element_type=acc)),
}


@pytest.mark.parametrize("kind", list(_KINDS))
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "int8", "uint8"])
def test_preferred_element_type(dtype, kind):
    cpu, mps = _devices()
    dtype = jnp.dtype(dtype)
    isint = jnp.issubdtype(dtype, jnp.integer)
    acc = jnp.int32 if isint else jnp.float32
    (sa, sb), make = _KINDS[kind]
    rng = np.random.default_rng(0)
    a, b = _arr(rng, sa, dtype), _arr(rng, sb, dtype)
    f = make(acc)
    want = np.asarray(jax.jit(f)(jax.device_put(a, cpu), jax.device_put(b, cpu)))
    got = np.asarray(jax.jit(f)(jax.device_put(a, mps), jax.device_put(b, mps)))
    assert got.dtype == want.dtype
    if isint:
        np.testing.assert_array_equal(got, want)
    else:
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-5 * scale)


def test_bf16_dot_default_result_stays_bf16():
    cpu, mps = _devices()
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 24)).astype(jnp.bfloat16)
    b = rng.standard_normal((24, 8)).astype(jnp.bfloat16)
    want = np.asarray(jax.jit(jnp.dot)(jax.device_put(a, cpu), jax.device_put(b, cpu)))
    got = np.asarray(jax.jit(jnp.dot)(jax.device_put(a, mps), jax.device_put(b, mps)))
    assert got.dtype == want.dtype == jnp.bfloat16
    np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=2e-2, atol=5e-2)


def test_bf16_grad_with_f32_accumulation():
    cpu, mps = _devices()
    rng = np.random.default_rng(2)
    w = rng.standard_normal((12, 5)).astype(jnp.bfloat16)
    x = rng.standard_normal((7, 12)).astype(jnp.bfloat16)

    def loss(w, x):
        y = lax.dot(x, w, preferred_element_type=jnp.float32)
        return jnp.sum(jnp.tanh(y))

    g = jax.grad(loss)
    want = np.asarray(jax.jit(g)(jax.device_put(w, cpu), jax.device_put(x, cpu))).astype(np.float32)
    got = np.asarray(jax.jit(g)(jax.device_put(w, mps), jax.device_put(x, mps))).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=5e-2)
