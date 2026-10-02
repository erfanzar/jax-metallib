# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The float32 logaddexp-pattern cut must not change any value.

jnp.logaddexp and jax.nn.softplus lower to
select(d != d, a + b, max(a, b) + log1p(exp(-|d|))). With HandleLogPlusOne's fan-out,
that segment exceeds MPSGraph's stitching budget once more work joins it (f32
[4096,4096] mish 12.1 ms, logaddexp 8.9, softplus(x) * softplus(-x) 43.5).
HandleLogPlusOne therefore materialises the log1p result with an int32 reinterpret
round trip (2.8 / 2.8 / 4.3 ms). The cut applies to f32 arrays of at least 2^21
elements, and not to a bare softplus whose result only reaches a return or reduce.
It must be bitwise identical to JAX_METALLIB_DISABLE_LOGADDEXP_CUT=1 in every dtype
and context. The flag is read once per process, so each route runs in a child.
"""

import os
import subprocess
import sys

import jax
import numpy as np
import pytest

try:
    mps_devices = jax.devices("mps")
except RuntimeError:
    mps_devices = []
pytestmark = pytest.mark.skipif(not mps_devices, reason="requires the mps backend")

_FLAG = "JAX_METALLIB_DISABLE_LOGADDEXP_CUT"

_CHILD = r"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

mps = jax.devices("mps")[0]
rng = np.random.default_rng(0)
base = (rng.standard_normal((2048, 1024)) * 3).astype(np.float32)
special = np.array([0.0, -0.0, 1e-30, -1e-30, 1.5e-45, 20.0, -20.0, 88.0, -88.0, 3e38, -3e38, np.inf, -np.inf,
                    np.nan], np.float32)
base[0, : special.size] = special
base[1, : special.size] = -special
# Weak-typed so f16/bf16 inputs (and the loop carry) stay in their own dtype.
H, ONE = 0.5, 1.0


def loop_body(x):
    def body(i, c):
        return jax.nn.softplus(c) * H - x

    return lax.fori_loop(0, 3, body, x)


CASES = {
    "mish": jax.nn.mish,
    "softplus": jax.nn.softplus,
    "softplus_add": lambda x: jax.nn.softplus(x) + H,
    "log_sigmoid": jax.nn.log_sigmoid,
    "logaddexp": lambda x: jnp.logaddexp(x, x * H),
    "logaddexp_c1": lambda x: jnp.logaddexp(x, ONE),
    "two_softplus": lambda x: jax.nn.softplus(x) * jax.nn.softplus(-x),
    "sum_softplus": lambda x: jnp.sum(jax.nn.softplus(x), axis=-1),
    "sum_2softplus": lambda x: jnp.sum(2 * jax.nn.softplus(x), axis=-1),
    "loop": loop_body,
}

out = {}
for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
    x = jax.device_put(jnp.asarray(base).astype(dtype), mps)
    for name, f in CASES.items():
        y = np.asarray(jax.jit(f)(x))
        out[f"{np.dtype(dtype).name}_{name}"] = y.view({4: np.int32, 2: np.int16}[y.dtype.itemsize])
# Rank-0 and small inputs take the no-cut path; they must still run.
out["scalar"] = np.asarray(jax.jit(jax.nn.mish)(jnp.float32(-1.5))).reshape(1).view(np.int32)
np.savez(sys.argv[1], **out)
print("CHILD_OK")
"""


def _run(tmp_path, disabled):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    env.pop(_FLAG, None)
    if disabled:
        env[_FLAG] = "1"
    path = tmp_path / ("off.npz" if disabled else "on.npz")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, (proc.stdout + proc.stderr)[-4000:]
    return np.load(path)


def test_logaddexp_cut_is_bitwise_identical(tmp_path):
    on = _run(tmp_path, disabled=False)
    off = _run(tmp_path, disabled=True)
    assert sorted(on.files) == sorted(off.files)
    diffs = {k: int((on[k] != off[k]).sum()) for k in on.files}
    assert all(d == 0 for d in diffs.values()), {k: d for k, d in diffs.items() if d}
    assert len(on.files) == 31
