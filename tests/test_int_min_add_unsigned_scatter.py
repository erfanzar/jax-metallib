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
"""Integer wraparound edges that silently miscompiled.

1. jit `x + int32(-2**31)` (a splat constant equal to the signed minimum, also
   int64) returned x unchanged wherever the sum overflows. Lowered as x ^ MIN.
2. Unsigned scatter reducers on a 1-D operand (x.at[i].add/max/min/mul) went
   to MPSGraph scatter directly: adds/muls saturated (INT32_MAX, 65535) and
   uint32 compared as signed. They now run in int32.
Each case runs in a subprocess so a regression fails one case.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import sys
    import jax
    name = sys.argv[1]
    if name.endswith("_i64"):
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, numpy as np
    cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
    U = np.uint32
    iv = np.array([-294967296, 5, -1294967296, 2147483647, -1, 0, -2147483648, 1], np.int32)
    MIN = np.int32(-2147483648)
    uv = np.array([4000000000, 5, 3000000000, 7], np.uint32)
    small = np.array([10, 5, 20, 7], np.uint32)
    u16 = np.array([60000, 5, 65535, 1], np.uint16)
    u8 = np.array([250, 5, 255, 1], np.uint8)
    cases = {
        "add_min": (lambda a: a + MIN, (iv,)),
        "min_plus_a": (lambda a: MIN + a, (iv,)),
        "add_min_2d": (lambda a: a.reshape(2, 4) + MIN, (iv,)),
        "add_min_then_mul": (lambda a: (a + MIN) * np.int32(3), (iv,)),
        "add_min_plus1": (lambda a: a + np.int32(-2147483647), (iv,)),
        "add_min_arg": (lambda a, b: a + b, (iv, np.full(8, -2147483648, np.int32))),
        "add_min_i16": (lambda a: a.astype(jnp.int16) + np.int16(-32768), (iv,)),
        "add_min_i64": (lambda a: a.astype(jnp.int64) + np.int64(-9223372036854775808), (iv,)),
        "u32_add_wrap": (lambda a, u: a.at[1].add(u), (uv, U(4294967295))),
        "u32_add_bigres": (lambda a, u: a.at[1].add(u), (small, U(3000000000))),
        "u32_add_bigin": (lambda a, u: a.at[0].add(u), (uv, U(3))),
        "u32_max_big": (lambda a, u: a.at[1].max(u), (small, U(4100000000))),
        "u32_max_keep": (lambda a, u: a.at[0].max(u), (uv, U(3))),
        "u32_min_big": (lambda a, u: a.at[0].min(u), (uv, U(3500000000))),
        "u32_mul_wrap": (lambda a, u: a.at[1].mul(u), (uv, U(1000000000))),
        "u32_set_big": (lambda a, u: a.at[1].set(u), (uv, U(4100000000))),
        "u32_dyn_idx": (lambda a, u, i: a.at[i].max(u), (uv, U(4100000000), np.int32(2))),
        "u32_vec_add": (lambda a, u: a.at[jnp.array([0, 2, 0])].add(u), (uv, np.array([4e9, 3e9, 1e9], np.uint32))),
        "u32_2d_add": (lambda a, u: a.reshape(2, 2).at[1, 0].add(u), (uv, U(4294967295))),
        "u16_add_wrap": (lambda a, u: a.at[1].add(u), (u16, np.uint16(65535))),
        "u16_max": (lambda a, u: a.at[1].max(u), (u16, np.uint16(65000))),
        "u16_mul_wrap": (lambda a, u: a.at[0].mul(u), (u16, np.uint16(60000))),
        "u8_add_wrap": (lambda a, u: a.at[0].add(u), (u8, np.uint8(10))),
        "u8_min": (lambda a, u: a.at[2].min(u), (u8, np.uint8(200))),
    }
    f, args = cases[name]
    want = np.asarray(jax.jit(f)(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(a, mps) for a in args]))
    assert got.shape == want.shape and got.dtype == want.dtype, (got.dtype, want.dtype)
    assert got.tobytes() == want.tobytes(), (got.ravel().tolist(), want.ravel().tolist())
    print("CASE_OK")
    """
)

_CASES = [
    "add_min",
    "min_plus_a",
    "add_min_2d",
    "add_min_then_mul",
    "add_min_plus1",
    "add_min_arg",
    "add_min_i16",
    "add_min_i64",
    "u32_add_wrap",
    "u32_add_bigres",
    "u32_add_bigin",
    "u32_max_big",
    "u32_max_keep",
    "u32_min_big",
    "u32_mul_wrap",
    "u32_set_big",
    "u32_dyn_idx",
    "u32_vec_add",
    "u32_2d_add",
    "u16_add_wrap",
    "u16_max",
    "u16_mul_wrap",
    "u8_add_wrap",
    "u8_min",
]


@pytest.mark.parametrize("name", _CASES)
def test_integer_wraparound_matches_cpu(name):
    import jax

    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _CHILD, name], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "CASE_OK" in r.stdout, f"rc={r.returncode}\n{r.stderr[-3000:]}"
