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
"""threefry2x32_p (the default jax.random PRNG) on MPS must match XLA:CPU bit for bit.

``jax_plugins/silicon/random_lowerings.py`` lowers threefry2x32_p for MPS with an int32
reinterpret after each round, so MPSGraph fuses short pieces instead of running the ~100-op
uint32 chain unfused (jax.random.bits 2^24: 154.9 -> 90.0 ms). The reinterpret must not change
any bit, and the rule must keep JAX's broadcasting and empty-output behaviour. Runs in a child
process so a native abort fails the test, not the suite.
"""

import os
import subprocess
import sys

import pytest

CHILD = r"""
import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.random import threefry2x32 as tf

cpu, mps = jax.devices("cpu")[0], jax.devices("mps")[0]
bad = []
n = 0


def check(tag, jf, *args):
    global n
    n += 1
    try:
        c, m = (
            [np.asarray(r) for r in jax.tree_util.tree_leaves(jf(*[jax.device_put(a, d) for a in args]))]
            for d in (cpu, mps)
        )
    except Exception as e:  # noqa: BLE001
        bad.append(f"{tag}: {type(e).__name__}: {str(e)[:120]}")
        return
    if len(c) != len(m) or not all(
        a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b) for a, b in zip(c, m)
    ):
        bad.append(tag)


# The MPS rule must be the one in use (not JAX's generic unrolled rule).
from jax._src.interpreters import mlir
rules = mlir._platform_specific_lowerings.get("mps", {})
print("MPS_RULE", tf.threefry2x32_p in rules)

kd = [np.asarray(jax.random.key_data(jax.random.key(s))) for s in (0, 7, 2**31 - 1)]
for i, k in enumerate(kd):
    for shape in [(), (0,), (1,), (3,), (7, 5), (4, 129), (2, 0, 3), (1000,), (1 << 16,)]:
        f = jax.jit(lambda d, s=shape: jax.random.bits(jax.random.wrap_key_data(d), s))
        check(f"bits/{shape}/key{i}", f, k)
    for name, fn in (
        ("uniform", lambda key: jax.random.uniform(key, (64, 33))),
        ("randint", lambda key: jax.random.randint(key, (257,), -5, 1000)),
        ("bits_u8", lambda key: jax.random.bits(key, (31,), jnp.uint8)),
        ("bits_u16", lambda key: jax.random.bits(key, (9, 3), jnp.uint16)),
        ("split", lambda key: jax.random.key_data(jax.random.split(key, 5))),
        ("fold_in", lambda key: jax.random.key_data(jax.random.fold_in(key, 12345))),
        ("permutation", lambda key: jax.random.permutation(key, 100)),
        ("split_vmap", lambda key: jax.vmap(lambda s: jax.random.bits(s, (9,)))(jax.random.split(key, 4))),
        ("bernoulli", lambda key: jax.random.bernoulli(key, 0.3, (513,))),
    ):
        f = jax.jit(lambda d, fn=fn: fn(jax.random.wrap_key_data(d)))
        check(f"{name}/key{i}", f, k)

    # Keys threaded through a scan on device must give the same stream.
    def scan_fn(d):
        def body(key, _):
            key, sub = jax.random.split(key)
            return key, jax.random.bits(sub, (6,))

        key, out = lax.scan(body, jax.random.wrap_key_data(d), None, length=5)
        return jax.random.key_data(key), out

    check(f"scan/key{i}", jax.jit(scan_fn), k)

# Direct primitive calls: operand broadcasting (size-1 dims), uint32 extremes.
u = np.uint32
x = np.arange(24, dtype=u).reshape(4, 6) * u(0x9E3779B9)
check("direct/full", jax.jit(lambda a, b, c, d: tf.threefry_2x32((a, b), jnp.stack([c, d]))),
      u(0xFFFFFFFF), u(0), x.ravel()[:12], x.ravel()[12:])
# JAX's CPU rule (rolled fori_loop) cannot take keys of different shapes: rotating the key
# schedule changes the loop-carry type. So CPU gets the operands pre-broadcast (same result by the
# primitive's broadcasting semantics), and MPS gets the raw (1,6)/(4,1) keys.
n += 1
bargs = (np.full((1, 6), 0xDEADBEEF, u), np.full((4, 1), 0x12345678, u), x, x[::-1].copy())
bind = jax.jit(lambda k1, k2, x1, x2: tf.threefry2x32_p.bind(k1, k2, x1, x2))
try:
    c = [np.asarray(r) for r in bind(*[jax.device_put(np.broadcast_to(a, (4, 6)).copy(), cpu) for a in bargs])]
    m = [np.asarray(r) for r in bind(*[jax.device_put(a, mps) for a in bargs])]
    if not all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(c, m)):
        bad.append("direct/bcast")
except Exception as e:  # noqa: BLE001
    bad.append(f"direct/bcast: {type(e).__name__}: {str(e)[:120]}")
check("direct/empty", jax.jit(lambda k1, k2, x1, x2: tf.threefry2x32_p.bind(k1, k2, x1, x2)),
      *[np.zeros((3, 0), u)] * 4)

print(f"CHECKED={n} BAD={len(bad)}")
for tag in bad[:20]:
    print("BAD", tag)
print("TF_OK" if not bad else "TF_FAIL")
sys.exit(1 if bad else 0)
"""


def test_threefry_round_cut_matches_cpu():
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_ENABLE_X64="0")
    proc = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("MPS_RULE", "CHECKED", "BAD", "TF_"))]
    print("\n".join(lines))
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "MPS_RULE True" in proc.stdout
    assert "TF_OK" in proc.stdout


@pytest.mark.parametrize("opt_out", ["JAX_METALLIB_DISABLE_THREEFRY_ROUND_CUT"])
def test_threefry_round_cut_opt_out(opt_out):
    code = (
        "import jax\n"
        "from jax._src.interpreters import mlir\n"
        "from jax._src.random import threefry2x32 as tf\n"
        "jax.devices('mps')\n"
        "print('MPS_RULE', tf.threefry2x32_p in mlir._platform_specific_lowerings.get('mps', {}))\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", **{opt_out: "1"})
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "MPS_RULE False" in proc.stdout
