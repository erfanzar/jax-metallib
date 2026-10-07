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

"""[..., K] @ [K, N] dots without batch dims (dense layers on sequences).

These lower to a broadcasting MPSGraph matmul instead of flattening the lhs to
[prod(...), K]. Results must match XLA:CPU for every rank and dtype.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]

SHAPES = [
    ((16, 256, 512), (512, 2048)),
    ((16, 256, 2048), (512,)),  # replaced below: rhs needs rank 2
    ((2, 3, 7, 33), (33, 5)),
    ((3, 1, 64), (64, 1)),
    ((1, 1, 1), (1, 8)),
    ((4, 128, 96), (96, 384)),
]
SHAPES[1] = ((16, 256, 2048), (2048, 512))

TOL = {np.float32: 2e-5, np.float16: 5e-3, jnp.bfloat16: 2e-2}


def _run(fn, *args, device):
    return np.asarray(jax.jit(fn)(*(jax.device_put(a, device) for a in args))).astype(np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16], ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize(("lshape", "rshape"), SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rankn_dot_matches_cpu(lshape, rshape, dtype):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(lshape).astype(np.float32).astype(dtype)
    b = (rng.standard_normal(rshape) * 0.05).astype(np.float32).astype(dtype)

    def fn(x, y):
        return x @ y

    out, ref = _run(fn, a, b, device=MPS), _run(fn, a, b, device=CPU)
    assert out.shape == ref.shape == (*lshape[:-1], rshape[1])
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    assert float(np.max(np.abs(out - ref))) / scale <= TOL[dtype]


# [..., K] x [N, K] (contract last x last), the input gradient g @ W.T of a
# dense layer. With prod(lhs free dims) == N the old flatten lowering aborted the
# process in MPSGraph canonicalization; the 31/33 cases are the M != N controls.
NT_SHAPES = [
    ((2, 16, 8), (32, 8)),
    ((4, 8, 8), (32, 8)),
    ((2, 8, 8), (16, 8)),
    ((8, 4, 16), (32, 16)),
    ((2, 16, 64), (32, 64)),
    ((2, 2, 16, 8), (64, 8)),
    ((2, 16, 8), (31, 8)),
    ((2, 16, 8), (33, 8)),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
@pytest.mark.parametrize(("lshape", "rshape"), NT_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rankn_dot_transposed_rhs_matches_cpu(lshape, rshape, dtype):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(lshape).astype(dtype)
    b = (rng.standard_normal(rshape) * 0.05).astype(dtype)
    dims = (((len(lshape) - 1,), (1,)), ((), ()))

    def fn(x, y):
        return jax.lax.dot_general(x, y, dims)

    out, ref = _run(fn, a, b, device=MPS), _run(fn, a, b, device=CPU)
    assert out.shape == ref.shape == (*lshape[:-1], rshape[0])
    assert float(np.max(np.abs(out - ref))) / float(np.max(np.abs(ref))) <= TOL[dtype]


def test_rankn_dot_preferred_f32_from_f16():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 64, 128)).astype(np.float16)
    b = rng.standard_normal((128, 32)).astype(np.float16)

    def fn(x, y):
        return jax.lax.dot_general(x, y, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    out, ref = _run(fn, a, b, device=MPS), _run(fn, a, b, device=CPU)
    assert float(np.max(np.abs(out - ref))) / float(np.max(np.abs(ref))) <= 2e-5


def test_rankn_dot_int32_exact():
    rng = np.random.default_rng(2)
    a = rng.integers(-50, 50, size=(3, 5, 17), dtype=np.int32)
    b = rng.integers(-50, 50, size=(17, 6), dtype=np.int32)
    out = np.asarray(jax.jit(jnp.matmul)(jax.device_put(a, MPS), jax.device_put(b, MPS)))
    np.testing.assert_array_equal(out, a @ b)


def test_rankn_dot_grad_matches_cpu():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 16, 32)).astype(np.float32)
    b = rng.standard_normal((32, 8)).astype(np.float32)

    def loss(x, y):
        return jnp.sum(jnp.tanh(x @ y))

    g = jax.grad(loss, argnums=(0, 1))
    for gm, gc in zip(
        jax.jit(g)(jax.device_put(a, MPS), jax.device_put(b, MPS)),
        jax.jit(g)(jax.device_put(a, CPU), jax.device_put(b, CPU)),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(gm), np.asarray(gc), rtol=1e-4, atol=1e-5)
