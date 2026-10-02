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
"""MPSGraph padTensor rounded 32-bit integer OPERANDS through f32.

1. stablehlo.pad with a constant pad value that f32 represents exactly (0, -1,
   INT32_MIN) took the native padTensor route, which returned every int32 above
   2^24 rounded to f32 (1638082997 -> 1638082944). Pad values f32 cannot
   represent (INT32_MAX, iinfo.max - 2) already fell back to the exact path, so
   the older pad test never reached it. This also broke shift-and-max scans
   written with lax.pad.
2. Integer convolution with lhs_dilation used a rank-6 padTensor and returned
   ~99% of int32 outputs wrong for |x| > 2^24.
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
    import jax, jax.numpy as jnp, numpy as np
    from jax import lax
    cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
    name, dts = sys.argv[1].split(":")
    dt = {"i32": np.int32, "u32": np.uint32, "i16": np.int16, "u8": np.uint8}[dts]
    info = np.iinfo(dt)
    rng = np.random.default_rng(5)
    x = rng.integers(info.min, info.max, size=(64, 256), endpoint=True, dtype=np.int64).astype(dt)
    x[0, :2] = [info.min, info.max]
    lo, neg1 = np.array(info.min, dt), np.array(-1 if info.min < 0 else 0, dt)
    z = np.array(0, dt)
    def conv(pad, lhs_dil):
        def f(a):
            k = (jnp.arange(3 * 3 * 2 * 3, dtype=dt).reshape(3, 3, 2, 3) % 3).astype(dt)
            return lax.conv_general_dilated(a[:2 * 9 * 11 * 2].reshape(2, 9, 11, 2), k, (1, 1), pad,
                                            lhs_dilation=lhs_dil, dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return lambda v: f(v.ravel())
    cases = {
        "pad_zero": lambda v: lax.pad(v, z, [(0, 0, 0), (3, 2, 0)]),
        "pad_neg1": lambda v: lax.pad(v, neg1, [(1, 0, 0), (0, 3, 0)]),
        "pad_min": lambda v: lax.pad(v, lo, [(0, 2, 0), (3, 0, 0)]),
        "shift_max": lambda v: jnp.maximum(v, lax.pad(v[:, :-1], lo, [(0, 0, 0), (1, 0, 0)])),
        "jnp_pad": lambda v: jnp.pad(v, ((2, 1), (0, 4)), constant_values=neg1),
        "conv_lhsdil": conv([(0, 0), (0, 0)], (2, 2)),
        "conv_lhsdil_pad": conv([(1, 1), (1, 1)], (2, 2)),
        "conv_pad": conv([(2, 0), (0, 3)], (1, 1)),
    }
    f = jax.jit(cases[name])
    want = np.asarray(f(jax.device_put(x, cpu)))
    got = np.asarray(f(jax.device_put(x, mps)))
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape, got.dtype, want.dtype)
    if got.tobytes() != want.tobytes():
        bad = np.flatnonzero(got.ravel() != want.ravel())
        raise SystemExit(f"MISMATCH n={bad.size}/{got.size} got={got.ravel()[bad[:3]]} want={want.ravel()[bad[:3]]}")
    print("CASE_OK")
    """
)

_CASES = [f"{n}:{d}" for n in ("pad_zero", "pad_neg1", "pad_min", "shift_max", "jnp_pad") for d in ("i32", "u32")]
_CASES += [f"{n}:{d}" for n in ("pad_neg1", "pad_min") for d in ("i16", "u8")]
_CASES += [f"{n}:{d}" for n in ("conv_lhsdil", "conv_lhsdil_pad", "conv_pad") for d in ("i32", "u32")]


def _run(case: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    return subprocess.run([sys.executable, "-c", _CHILD, case], capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.parametrize("case", _CASES)
def test_int_pad_exact(case):
    r = _run(case)
    tail = (r.stdout + r.stderr)[-1500:]
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), tail
