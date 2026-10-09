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

"""Flat (1-D) elementwise segments that contain broadcast scalar constants.

A rank-0 float constant broadcast to the segment shape (x * 1.702, t * 0.1)
no longer disqualifies the flat path: the broadcast is bound to a [1] tensor
that MPSGraph broadcasts implicitly. Results must match XLA:CPU bit for bit
for add/sub/mul/max/min and to tolerance for transcendental chains.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]

DTYPES = [np.float32, np.float16, jnp.bfloat16]
IDS = ["f32", "f16", "bf16"]


def _run(fn, *args, device):
    return jax.jit(fn)(*(jax.device_put(x, device) for x in args))


def _bits(x):
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def _data(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float32)
    a.flat[::9] = -0.0
    a.flat[4] = np.inf
    return a.astype(dtype)


EXACT_OPS = {
    "mul_const": lambda x, y: x * 1.702,
    "add_const": lambda x, y: x + 0.25,
    "const_sub": lambda x, y: 3.0 - x,
    "max_const": lambda x, y: jnp.maximum(x, 0.5),
    "min_neg_zero": lambda x, y: jnp.minimum(x, -0.0),
    "two_consts": lambda x, y: (x * 2.0 + y) * 0.5,
}


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
@pytest.mark.parametrize("shape", [(64, 96), (4, 8, 33)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("op", list(EXACT_OPS))
def test_scalar_operand_bitwise(op, shape, dtype):
    x, y = _data(shape, dtype, 0), _data(shape, dtype, 1)
    fn = EXACT_OPS[op]
    out, ref = _run(fn, x, y, device=MPS), _run(fn, x, y, device=CPU)
    assert out.shape == ref.shape == shape
    np.testing.assert_array_equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
def test_gelu_and_chain_match_cpu(dtype):
    x, y, z, w = (_data((128, 256), dtype, s) * dtype(0.5) for s in range(4))

    def gelu(a):
        return a * jax.nn.sigmoid(dtype(1.702) * a)

    def chain5(a, b, c, d):
        return jnp.exp((jnp.tanh(a * b) + c) * dtype(0.1)) - d

    tol = 2e-6 if dtype == np.float32 else 2e-3
    for fn, args in ((gelu, (x,)), (chain5, (x, y, z, w))):
        out = np.asarray(_run(fn, *args, device=MPS), np.float64)
        ref = np.asarray(_run(fn, *args, device=CPU), np.float64)
        assert out.shape == ref.shape
        np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
        np.testing.assert_allclose(out, ref, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_shared_constant_mul_add(dtype):
    # x * c + y * c: one constant feeding two broadcasts. A mul+add pair may be
    # contracted to an FMA on either backend (1-ulp differences with the flat
    # path on or off), so compare to a tolerance rather than bitwise.
    x, y = _data((64, 96), dtype, 7), _data((64, 96), dtype, 8)

    def fn(a, b):
        return a * 0.1 + b * 0.1

    out = np.asarray(_run(fn, x, y, device=MPS), np.float64)
    ref = np.asarray(_run(fn, x, y, device=CPU), np.float64)
    tol = 1e-6 if dtype == np.float32 else 1e-2
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol, equal_nan=True)


def test_constant_also_used_outside_segment():
    # The scalar also feeds a row reduce, so its users leave the segment.
    x = _data((64, 4096), np.float32, 5)

    def fn(a):
        c = jnp.float32(0.5)
        return a * c, jnp.sum(a, axis=-1) * c

    out = _run(fn, x, device=MPS)
    ref = _run(fn, x, device=CPU)
    np.testing.assert_array_equal(_bits(out[0]), _bits(ref[0]))
    np.testing.assert_allclose(np.asarray(out[1]), np.asarray(ref[1]), rtol=2e-5, atol=1e-3)


def test_scalar_only_result_is_not_flattened():
    # An op whose operands are all broadcast scalars must keep its full shape.
    x = _data((32, 48), np.float32, 6)

    def fn(a):
        b = jnp.broadcast_to(jnp.float32(2.0), a.shape) * jnp.float32(3.0)
        return a + b

    out = _run(fn, x, device=MPS)
    ref = _run(fn, x, device=CPU)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(_bits(out), _bits(ref))
