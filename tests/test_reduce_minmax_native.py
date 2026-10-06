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
"""Bitwise last-axis max/min at sizes that take the native row kernel.

The oracle follows XLA semantics: NaN propagates, -0 orders below +0, and
float32 subnormals compare as a zero of the same sign (XLA:CPU and the
MPSGraph path both flush them). Shapes are above the 32768-element native
threshold; smaller edge tests never reach the kernel. NumPy is the oracle
because CPU YNN REDUCE misorders these edges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _has_mps():
    try:
        return bool(jax.devices("mps"))
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _has_mps(), reason="needs the mps backend")


def _keys(x):
    if x.dtype == np.float32:
        b = x.view(np.int32).astype(np.int64)
        b = np.where((b & 0x7F800000) == 0, b & -0x80000000, b)
        return np.where(b < 0, b ^ 0x7FFFFFFF, b)
    b = x.view(np.int16).astype(np.int64)
    return np.where(b < 0, b ^ 0x7FFF, b)


def _oracle(x, op):
    keys = _keys(x)
    idx = (np.argmax if op == "max" else np.argmin)(keys, axis=-1)
    out = np.take_along_axis(x, idx[:, None], axis=-1)[:, 0].copy()
    if x.dtype == np.float32:
        bits = out.view(np.int32)
        flush = (bits & 0x7F800000) == 0
        bits[flush] &= np.int32(-0x80000000)
    out[np.isnan(x).any(axis=-1)] = np.nan
    return out


def _make(dtype, shape):
    x = np.random.default_rng(7).standard_normal(shape).astype(dtype)
    sub = np.finfo(dtype).smallest_subnormal
    x[0] = -0.0
    x[1] = -0.0
    x[1, 7] = 0.0
    x[2] = 0.0
    x[2, 5] = -0.0
    x[3] = -np.abs(x[3])
    x[3, 11] = -0.0
    x[4] = -np.abs(x[4])
    x[4, 3] = -sub
    x[5] = np.abs(x[5])
    x[5, 9] = sub
    x[6, 100] = np.nan
    x[7, -1] = np.nan
    x[8, 0] = np.inf
    x[9, 0] = -np.inf
    x[10] = -np.inf
    x[11] = np.inf
    x[12] = sub
    x[12, 4] = 2 * sub
    x[13] = -sub
    x[13, 4] = -2 * sub
    x[14] = sub
    x[14, 6] = -0.0
    x[15] = -sub
    x[15, 6] = 0.0
    x[16::7, 13] = np.nan
    return x


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("shape", [(2048, 1024), (64, 65536), (512, 3000), (32, 400)])
@pytest.mark.parametrize("op", ["max", "min"])
def test_row_minmax_bitwise(dtype, shape, op):
    x = _make(dtype, shape)
    fn = jnp.max if op == "max" else jnp.min
    mps = jax.devices("mps")[0]
    got = np.asarray(jax.jit(lambda a: fn(a, axis=-1))(jax.device_put(x, mps)))
    ref = _oracle(x, op)
    itype = np.uint32 if dtype == np.float32 else np.uint16
    diff = (ref.view(itype) != got.view(itype)) & ~(np.isnan(ref) & np.isnan(got))
    assert not diff.any(), f"rows {np.nonzero(diff)[0][:8].tolist()} ref={ref[diff][:4]} got={got[diff][:4]}"
