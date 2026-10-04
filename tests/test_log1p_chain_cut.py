# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The float32 log1p chain cut must not change any value.

HandleLogPlusOne reads its input ~8 times, so two chained log1p ops exceed
MPSGraph's stitching budget (f32 [4096,4096] log1p(log1p(x)) 15.8 ms). When the
result reaches another log1p through elementwise ops, the handler forces it to
materialise with an int32 reinterpret round trip (2.2 ms). That cut is f32-only
and must be bitwise identical to JAX_METALLIB_DISABLE_LOG1P_CHAIN_CUT=1 in every
dtype and context: chains, softplus, log1p -> tanh, loop bodies and rank-0
inputs. The flag is read once per process, so each route runs in a child.
Empty inputs are not covered here: any jitted elementwise op on a (0, 4) array
aborts on MPS with or without the cut (separate, pre-existing bug).
"""

import os
import subprocess
import sys

import jax
import numpy as np
import pytest
from jax import numpy as jnp

try:
    mps_devices = jax.devices("mps")
except RuntimeError:
    mps_devices = []
pytestmark = pytest.mark.skipif(not mps_devices, reason="requires the mps backend")

_FLAG = "JAX_METALLIB_DISABLE_LOG1P_CHAIN_CUT"

_CHILD = r"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

mps = jax.devices("mps")[0]
rng = np.random.default_rng(0)
special = np.array([0.0, -0.0, 1e-30, 1.5e-45, -1e-8, 1e-4, -0.5, -0.999, -1.0, -2.0, 3e38, np.inf, -np.inf,
                    np.nan], np.float32)
xp = rng.uniform(0.0, 2.0, (256, 256)).astype(np.float32)
xp[0, : special.size] = special
xs = (rng.standard_normal((256, 256)) * 3).astype(np.float32)
xs[0, : special.size] = special


def chain(n):
    def f(x):
        for _ in range(n):
            x = jnp.log1p(x)
        return x

    return f


CASES = {
    "log1p1": (chain(1), xp),
    "log1p2": (chain(2), xp),
    "log1p4": (chain(4), xp),
    "scaled_chain": (lambda x: jnp.log1p(jnp.log1p(x) * np.float32(0.5) + np.float32(0.25)), xp),
    "softplus": (jax.nn.softplus, xs),
    "log1p_tanh": (lambda x: jnp.tanh(jnp.log1p(x) * np.float32(0.5)), xp),
    "cheap_after": (lambda x: jnp.log1p(x) * np.float32(1.5) + np.float32(0.25), xp),
    "fori_loop": (lambda x: lax.fori_loop(0, 3, lambda i, v: jnp.log1p(jnp.log1p(v)) * 1.1, x), xp),
    "rank0": (chain(3), np.float32(0.3)),
}

out = {}
for dt in (np.float32, np.float16, jnp.bfloat16):
    for name, (f, src) in CASES.items():
        a = np.asarray(jax.jit(f)(jax.device_put(np.asarray(src).astype(dt), mps)))
        out[f"{np.dtype(dt).name}_{name}"] = a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint16)
np.savez(sys.argv[1], **out)
print("DUMP_OK", len(out))
"""


def _dump(tmp_path, opt_out):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    env.pop(_FLAG, None)
    if opt_out:
        env[_FLAG] = "1"
    path = tmp_path / ("off.npz" if opt_out else "on.npz")
    r = subprocess.run([sys.executable, "-c", _CHILD, str(path)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "DUMP_OK" in r.stdout, f"rc={r.returncode}\n{r.stderr[-3000:]}"
    return np.load(path)


def test_log1p_chain_cut_is_bitwise_identical_to_opt_out(tmp_path):
    on = _dump(tmp_path, opt_out=False)
    off = _dump(tmp_path, opt_out=True)
    assert sorted(on.files) == sorted(off.files)
    diffs = {k: int((on[k] != off[k]).sum()) for k in on.files if on[k].shape == off[k].shape}
    shapes = {k for k in on.files if on[k].shape != off[k].shape}
    assert not shapes, f"shape mismatch: {sorted(shapes)}"
    assert not any(diffs.values()), {k: v for k, v in diffs.items() if v}


@pytest.mark.parametrize("n", [2, 4])
def test_log1p_chain_f32_close_to_cpu(n):
    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:
        pytest.skip("cpu backend not available")

    def f(x):
        for _ in range(n):
            x = jnp.log1p(x)
        return x

    x = np.random.default_rng(n).uniform(0.0, 2.0, (512, 512)).astype(np.float32)
    got = np.asarray(jax.jit(f)(jax.device_put(x, mps_devices[0])))
    want = np.asarray(jax.jit(f)(jax.device_put(x, cpu)))
    # Single-op f32 log1p already differs from XLA:CPU by up to ~6e-4 relative
    # (log(1+x) for |x| > 1e-4); this guards against both routes being wrong
    # in the same gross way, not against that pre-existing gap.
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=0)
