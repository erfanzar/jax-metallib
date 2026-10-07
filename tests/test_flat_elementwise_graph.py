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

"""Pure elementwise graph segments run on 1-D placeholders.

A segment of same-shape float elementwise ops is built on [numel]
placeholders over the same contiguous buffers. Results must be unchanged, and
a slot that feeds both a flattened segment and a segment using the declared
shape in one plan must get a tensor-data wrapper of the right shape for each.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]

DTYPES = [np.float32, np.float16, jnp.bfloat16]
IDS = ["f32", "f16", "bf16"]


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float32)
    b = rng.standard_normal(shape).astype(np.float32)
    a.flat[::7] = -0.0
    b.flat[::11] = 0.0
    a.flat[3] = np.nan
    b.flat[5] = np.inf
    return a.astype(dtype), b.astype(dtype)


def _run(fn, *args, device):
    return jax.jit(fn)(*(jax.device_put(x, device) for x in args))


def _bits(x):
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


SINGLE_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": jnp.maximum,
    "min": jnp.minimum,
    "neg": lambda a, b: -a,
    "abs": lambda a, b: jnp.abs(a),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
@pytest.mark.parametrize("shape", [(64, 96), (4, 8, 33), (2, 3, 5, 7)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("op", list(SINGLE_OPS))
def test_single_op_bitwise_matches_cpu(op, shape, dtype):
    a, b = _inputs(shape, dtype, 0)
    fn = SINGLE_OPS[op]
    out, ref = _run(fn, a, b, device=MPS), _run(fn, a, b, device=CPU)
    assert out.shape == ref.shape == shape
    if op == "neg" and dtype != np.float32:
        # Default MPSGraph half negate writes +qNaN for every NaN (opt-in fix:
        # JAX_METALLIB_HALF_BIT_SIGN_OPS=1); it is not part of this change.
        nan = np.isnan(np.asarray(ref, np.float32))
        np.testing.assert_array_equal(np.isnan(np.asarray(out, np.float32)), nan)
        np.testing.assert_array_equal(_bits(out)[~nan], _bits(ref)[~nan])
        return
    np.testing.assert_array_equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
def test_chain_matches_cpu(dtype):
    a, b = _inputs((128, 256), dtype, 1)

    def fn(x, y):
        return jnp.tanh(jnp.abs(x) * y - jnp.maximum(x, y)) / (jnp.exp(-jnp.abs(y)) + jnp.abs(x))

    out = np.asarray(_run(fn, a, b, device=MPS), np.float64)
    ref = np.asarray(_run(fn, a, b, device=CPU), np.float64)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    tol = 2e-6 if dtype == np.float32 else 2e-3
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
def test_slot_shared_by_flat_and_shaped_segments(dtype):
    # x feeds the flattened x + y segment and, after the native row reduce,
    # a broadcasting segment that binds x with its declared [64, 4096] shape.
    # The shape matters: at [4, 2048] the reduce stays in the graph and the
    # whole function is one segment, so no slot is shared between segments.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 4096)).astype(dtype)
    y = rng.standard_normal((64, 4096)).astype(dtype)

    def fn(a, b):
        s = a + b
        r = jnp.sum(s, axis=-1)
        return s, a * r[:, None]

    out = _run(fn, x, y, device=MPS)
    ref = _run(fn, x, y, device=CPU)
    np.testing.assert_array_equal(_bits(out[0]), _bits(ref[0]))
    tol = 2e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(out[1], np.float64), np.asarray(ref[1], np.float64), rtol=tol, atol=tol * 64)
