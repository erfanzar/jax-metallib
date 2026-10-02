# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""bitcast_convert between element widths must give the StableHLO result shape.

reinterpretCastTensor scales the last dimension (u16[6,4,2] -> u32 gave 6x4x1, u32[6,4] -> u16 gave
6x8) where StableHLO drops or appends that axis. A bare bitcast still returned the right bytes, but
any consumer saw the wrong rank: add/shift/multiply/reverse aborted the process inside MPSGraph and a
rank-1 add broadcast [8,1] + [8] to [8,8] and returned wrong values. Each case runs in a child
process (an abort fails one case) and compares raw bytes with XLA:CPU.
"""

import os
import subprocess
import sys

import pytest

CASES = [
    "widen_u16_u32_bare",
    "widen_u16_u32_shl",
    "widen_u16_u32_add",
    "widen_u16_u32_rank1_add",
    "widen_u16_u32_scalar_add",
    "widen_u8_u32_add",
    "widen_u8_u16_add",
    "widen_u16_f32_mul",
    "widen_u16_u32_flip",
    "widen_u16_u32_sum",
    "widen_u32_u64_add",
    "narrow_u32_u16_bare",
    "narrow_u32_u16_add",
    "narrow_u32_u16_flip_last",
    "narrow_u32_u8_add",
    "narrow_f32_u16_shr",
    "narrow_scalar_u32_u16_add",
    "roundtrip_u16_u32_u16_add",
    "pairpack_f16_flip_last",
    "pairpack_bf16_flip_both",
    "same_f32_u32_add",
]


def _run_case(name: str) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    rng = np.random.default_rng(0)

    def bits(shape, dt):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, size=shape, dtype=dt, endpoint=True)

    def bc(x, dt):
        return lax.bitcast_convert_type(x, dt)

    def pairpack_flip(x, axes):
        n = x.shape[-1]
        u = bc(x, jnp.uint16).reshape(*x.shape[:-1], n // 2, 2)
        w = jnp.flip(bc(u, jnp.uint32), axes)
        if (x.ndim - 1) in axes:
            w = (w << 16) | (w >> 16)
        return bc(bc(w, jnp.uint16).reshape(x.shape), x.dtype)

    cases = {
        "widen_u16_u32_bare": (lambda x: bc(x, jnp.uint32), bits((6, 4, 2), np.uint16)),
        "widen_u16_u32_shl": (lambda x: bc(x, jnp.uint32) << 16, bits((6, 4, 2), np.uint16)),
        "widen_u16_u32_add": (lambda x: bc(x, jnp.uint32) + 1, bits((6, 4, 2), np.uint16)),
        "widen_u16_u32_rank1_add": (lambda x: bc(x, jnp.uint32) + 1, bits((8, 2), np.uint16)),
        "widen_u16_u32_scalar_add": (lambda x: bc(x, jnp.uint32) + 1, bits((2,), np.uint16)),
        "widen_u8_u32_add": (lambda x: bc(x, jnp.uint32) + 1, bits((6, 4, 4), np.uint8)),
        "widen_u8_u16_add": (lambda x: bc(x, jnp.uint16) + 1, bits((3, 5, 2), np.uint8)),
        "widen_u16_f32_mul": (lambda x: bc(bc(x, jnp.float32) * 1.0, jnp.uint32), bits((6, 4, 2), np.uint16)),
        "widen_u16_u32_flip": (lambda x: jnp.flip(bc(x, jnp.uint32), 1), bits((6, 4, 2), np.uint16)),
        "widen_u16_u32_sum": (lambda x: jnp.sum(bc(x, jnp.uint32), axis=1), bits((6, 4, 2), np.uint16)),
        "widen_u32_u64_add": (lambda x: bc(x, jnp.uint64) + 1, bits((3, 4, 2), np.uint32)),
        "narrow_u32_u16_bare": (lambda x: bc(x, jnp.uint16), bits((6, 4), np.uint32)),
        "narrow_u32_u16_add": (lambda x: bc(x, jnp.uint16) + 1, bits((6, 4), np.uint32)),
        "narrow_u32_u16_flip_last": (lambda x: jnp.flip(bc(x, jnp.uint16), 2), bits((6, 4), np.uint32)),
        "narrow_u32_u8_add": (lambda x: bc(x, jnp.uint8) + 1, bits((6, 4), np.uint32)),
        "narrow_f32_u16_shr": (lambda x: bc(bc(x, jnp.float32), jnp.uint16) >> 3, bits((5, 3), np.uint32)),
        "narrow_scalar_u32_u16_add": (lambda x: bc(x, jnp.uint16) + 1, bits((), np.uint32)),
        "roundtrip_u16_u32_u16_add": (lambda x: bc(bc(x, jnp.uint32), jnp.uint16) + 1, bits((6, 4, 2), np.uint16)),
        "pairpack_f16_flip_last": (lambda x: pairpack_flip(x, (1,)), bits((6, 8), np.uint16).view(np.float16)),
        "pairpack_bf16_flip_both": (
            lambda x: pairpack_flip(bc(x, jnp.bfloat16), (0, 1)),
            bits((5, 4), np.uint16),
        ),
        "same_f32_u32_add": (lambda x: bc(bc(x, jnp.float32), jnp.uint32) + 1, bits((6, 4), np.uint32)),
    }
    if name not in cases:
        raise SystemExit(f"unknown case {name}")
    fn, x = cases[name]

    outs = {}
    for plat in ("cpu", "mps"):
        dev = jax.devices(plat)[0]
        out = jax.jit(fn)(jax.device_put(x, dev))
        assert next(iter(out.devices())).platform == plat, out.devices()
        outs[plat] = np.asarray(out)
    c, m = outs["cpu"], outs["mps"]
    assert c.shape == m.shape and c.dtype == m.dtype, (c.shape, c.dtype, m.shape, m.dtype)
    assert np.array_equal(c.reshape(-1).view(np.uint8), m.reshape(-1).view(np.uint8)), (c, m)
    print("OK")


@pytest.mark.parametrize("case", CASES)
def test_bitcast_width_change_matches_cpu(case):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run(
        [sys.executable, __file__, case], capture_output=True, text=True, timeout=120, env=env, check=False
    )
    assert r.returncode == 0 and "OK" in r.stdout, (r.returncode, r.stdout[-400:], r.stderr[-800:])


if __name__ == "__main__":
    _run_case(sys.argv[1])
