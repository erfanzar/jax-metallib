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
"""Bool interior padding and bf16 rounding before discontinuous ops match CPU.

1. lax.pad of a bool operand with interior padding returned the padding value in
   every element (MPSGraph scatterND dropped bool updates; int8 was exact).
2. MPSGraph evaluated f16/bf16 elementwise chains without rounding intermediates
   to the half type, so round(x * 1.5) with bf16 x = 4.34375 gave 7 instead of 6
   (the product 6.515625 is a bf16 tie that rounds to 6.5), and f16
   x * 1.5 == 3.00390625 was false for x = 2.001953125. By default a half value
   feeding round/floor/ceil/sign/compare/rem is rounded;
   JAX_METALLIB_BF16_STRICT_ROUNDING=1 rounds every bf16 op result (XLA:CPU does;
   it keeps f16 arithmetic chains in f32, so strict mode leaves f16 alone).
"""

import os
import subprocess
import sys
import textwrap

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


def _run(f, *args):
    cpu, mps = _devices()
    want = np.asarray(jax.jit(f)(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(a, mps) for a in args]))
    assert got.dtype == want.dtype and got.shape == want.shape
    return want, got


_PAD_CFGS = [
    ((0, 0, 1), (0, 0, 0)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 1), (2, -1, 0)),
    ((0, 0, 1), (0, 0, 2)),
    ((1, 2, 0), (0, -1, 0)),
    ((2, -1, 3), (1, 1, 1)),
]


@pytest.mark.parametrize("pad_value", [True, False])
@pytest.mark.parametrize("cfg", _PAD_CFGS, ids=[str(c) for c in _PAD_CFGS])
def test_bool_pad_matches_cpu(cfg, pad_value):
    x = np.random.default_rng(0).random((5, 6)) > 0.5
    want, got = _run(lambda x: lax.pad(x, jnp.array(pad_value), cfg), x)
    np.testing.assert_array_equal(got, want)


def test_bool_pad_rank3_interior():
    x = np.random.default_rng(1).random((3, 4, 5)) > 0.3
    cfg = ((0, 1, 2), (1, 0, 1), (0, 0, 3))
    want, got = _run(lambda x: lax.pad(x, jnp.array(False), cfg), x)
    np.testing.assert_array_equal(got, want)


# 4.34375 * 1.5 = 6.515625 (exact in f32) is a bf16 tie that rounds to 6.5.
_X = np.array([4.34375, 1.0, 2.171875, -4.34375, 0.5, -2.171875], np.float32)

_ROUNDING_CASES = {
    "round": lambda x: jnp.round(x * 1.5),
    "floor": lambda x: jnp.floor(x * 1.5 + 0.5),
    "ceil": lambda x: jnp.ceil(x * 1.5 - 0.5),
    "compare": lambda x: x * 1.5 > 6.5,
    "where_compare": lambda x: jnp.where(x * 1.5 <= 6.5, x, -x),
    "rem": lambda x: lax.rem(x * 1.5, jnp.asarray(0.5, x.dtype)),
    "reshape_then_round": lambda x: jnp.round((x * 1.5).reshape(2, 3)),
}


@pytest.mark.parametrize("name", sorted(_ROUNDING_CASES))
def test_bf16_rounding_before_discontinuous_op(name):
    want, got = _run(_ROUNDING_CASES[name], _X.astype(jnp.bfloat16))
    np.testing.assert_array_equal(got.astype(np.float64), want.astype(np.float64))


# 2.001953125 * 1.5 = 3.0029296875 is an f16 tie that rounds to 3.00390625.
_X16 = np.array([2.001953125, 1.0, -2.001953125, 0.25, 7.0, -1.0], np.float32)
_F16_TIE = 3.00390625

_F16_CASES = {
    **_ROUNDING_CASES,
    "eq_tie": lambda x: x * 1.5 == jnp.asarray(_F16_TIE, x.dtype),
    "where_eq_tie": lambda x: jnp.where(x * 1.5 == jnp.asarray(_F16_TIE, x.dtype), x, -x),
    "floor_scaled": lambda x: jnp.floor(x * 1.5 * 256),
}


@pytest.mark.parametrize("name", sorted(_F16_CASES))
def test_f16_rounding_before_discontinuous_op(name):
    want, got = _run(_F16_CASES[name], _X16.astype(np.float16))
    np.testing.assert_array_equal(got.astype(np.float64), want.astype(np.float64))


_STRICT_CHILD = textwrap.dedent(
    """
    import jax, jax.numpy as jnp, numpy as np
    cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
    x = np.array([4.34375, 1.0, 2.171875, -4.34375], np.float32).astype(jnp.bfloat16)
    fs = [lambda x: (x * 1.5) - 6.5, lambda x: jnp.exp(x * 1.5), lambda x: jnp.sign(x * 1.5 - 6.5)]
    for f in fs:
        w = np.asarray(jax.jit(f)(jax.device_put(x, cpu))).astype(np.float64)
        m = np.asarray(jax.jit(f)(jax.device_put(x, mps))).astype(np.float64)
        assert np.array_equal(w, m), (w.tolist(), m.tolist())
    # f16 arithmetic chains stay unrounded on XLA:CPU; strict mode must not round them.
    h = np.array([2.001953125, 1.0], np.float32).astype(np.float16)
    c = np.float16(3.00390625)
    for f in [lambda x: x * 1.5 - c, lambda x: jnp.sign(x * 1.5 - c), lambda x: x * 1.5 == c]:
        w = np.asarray(jax.jit(f)(jax.device_put(h, cpu))).astype(np.float64)
        m = np.asarray(jax.jit(f)(jax.device_put(h, mps))).astype(np.float64)
        assert np.array_equal(w, m), (w.tolist(), m.tolist())
    print("STRICT_OK")
    """
)


def test_bf16_strict_rounding_matches_cpu_exactly():
    _devices()
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_BF16_STRICT_ROUNDING="1")
    r = subprocess.run([sys.executable, "-c", _STRICT_CHILD], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "STRICT_OK" in r.stdout, r.stderr[-2000:]
