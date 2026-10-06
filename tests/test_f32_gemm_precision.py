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

"""Large float32 matmuls must keep float32 inputs by default.

The graph dot handler used to cast both float32 operands to float16 once they
were large enough. That gave ~3e-4 relative error at normal scales and
non-finite results when products exceeded the float16 range (inputs ~1e5),
and it ignored precision=HIGHEST.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]


def _run(fn, *args, device):
    return np.asarray(jax.jit(fn)(*(jax.device_put(a, device) for a in args)))


def _rel(out, ref):
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e5])
@pytest.mark.parametrize("precision", [None, jax.lax.Precision.HIGHEST])
def test_large_f32_matmul_matches_cpu(scale, precision):
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((1024, 1024)) * scale).astype(np.float32)
    b = (rng.standard_normal((1024, 1024)) / scale).astype(np.float32)

    def fn(x, y):
        return jnp.matmul(x, y, precision=precision)

    out, ref = _run(fn, a, b, device=MPS), _run(fn, a, b, device=CPU)
    assert np.isfinite(out).all()
    assert _rel(out, ref) <= 2e-5


def test_large_f32_matmul_overflowing_f16_range():
    # Every product is 7e4 * 1 = 7e4 > float16 max (65504); exact in float32.
    a = np.full((1024, 1024), 7e4, dtype=np.float32)
    b = np.full((1024, 1024), 1.0 / 1024, dtype=np.float32)
    out = _run(jnp.matmul, a, b, device=MPS)
    np.testing.assert_allclose(out, 7e4, rtol=1e-6)


def test_batched_nt_f32_matmul_matches_cpu():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, 16, 256, 64)).astype(np.float32)
    k = rng.standard_normal((4, 16, 256, 64)).astype(np.float32)

    def fn(x, y):
        return x @ jnp.swapaxes(y, -2, -1)

    assert _rel(_run(fn, q, k, device=MPS), _run(fn, q, k, device=CPU)) <= 2e-5
