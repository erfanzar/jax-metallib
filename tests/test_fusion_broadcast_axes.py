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
"""Native softmax/LayerNorm/mean fusions must not fire when a last-axis
reduction is broadcast back along the wrong axis.

On a square input, x - max(x, axis=1)[None, :] has the shapes of a row softmax
but is not one. The fusion matchers traced the broadcast chain without checking
broadcast_dimensions and fused it (relative error up to ~6e2 vs CPU at n=1024).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _mps():
    try:
        return jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")


def softmax_wrong_max(x):
    m = jnp.max(x, axis=1)
    e = jnp.exp(x - m[None, :])
    return e / jnp.sum(e, axis=1, keepdims=True)


def softmax_wrong_sum(x):
    m = jnp.max(x, axis=1, keepdims=True)
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=1)
    return e / s[None, :]


def softmax_numerator_wrong(x):
    m = jnp.max(x, axis=1)
    return jnp.exp(x - m[None, :])


def softmax_right(x):
    return jax.nn.softmax(x, axis=-1)


def layernorm_wrong_mean(x):
    mu = jnp.mean(x, axis=1)
    d = x - mu[None, :]
    var = jnp.mean(d * d, axis=1, keepdims=True)
    return d / jnp.sqrt(var + 1e-5)


def layernorm_wrong_var(x):
    mu = jnp.mean(x, axis=1, keepdims=True)
    d = x - mu
    var = jnp.mean(d * d, axis=1)
    return d / jnp.sqrt(var[None, :] + 1e-5)


def layernorm_right(x):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    d = x - mu
    var = jnp.mean(d * d, axis=-1, keepdims=True)
    return d / jnp.sqrt(var + 1e-5)


def mean_wrong(x):
    return x - jnp.mean(x, axis=1)[None, :]


@pytest.mark.parametrize(
    "fn",
    [
        softmax_wrong_max,
        softmax_wrong_sum,
        softmax_numerator_wrong,
        softmax_right,
        layernorm_wrong_mean,
        layernorm_wrong_var,
        layernorm_right,
        mean_wrong,
    ],
)
@pytest.mark.parametrize("n", [64, 1024, 2048])
def test_fusion_respects_broadcast_axes(fn, n):
    mps = _mps()
    cpu = jax.devices("cpu")[0]
    x = (np.random.default_rng(n).standard_normal((n, n)) * 3).astype(np.float32)
    want = np.asarray(jax.jit(fn)(jax.device_put(x, cpu)))
    got = np.asarray(jax.jit(fn)(jax.device_put(x, mps)))
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)
