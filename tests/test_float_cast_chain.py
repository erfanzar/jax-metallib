# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Narrowing float conversions must round even when widened again.

Under jit, MPSGraph folded cast chains such as x.astype(f16).astype(f32) into
the identity, so 70000 stayed 70000 instead of inf and pi kept its float32
bits. f16 arithmetic between the casts (x.astype(f16) * 2) was also computed
in float32. HandleConvert now puts a reinterpret round trip on the 16-bit side
of each conversion, which MPSGraph cannot fold.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

f16, bf16, f32 = jnp.float16, jnp.bfloat16, jnp.float32

# bf16 subnormals are excluded: MPS flushes them to zero (known limitation).
_F32 = np.array(
    [70000.0, 65520.0, 40000.0, 3.14159265, 1.0 + 2**-12, 1e-6, 1e-8, -2.5, -0.0, 3.0e38, np.inf, np.nan],
    dtype=np.float32,
)
_INT = np.array([70000, 65520, 2049, 4097, -3, 0, 123456789], dtype=np.int32)

_CHAINS = {
    "f32_f16_f32": (lambda a: a.astype(f16).astype(f32), _F32),
    "f32_bf16_f32": (lambda a: a.astype(bf16).astype(f32), _F32),
    "f32_f16_mul_f32": (lambda a: (a.astype(f16) * f16(2)).astype(f32), _F32),
    # XLA:CPU keeps this bf16 product in f32 (xla_allow_excess_precision), so the
    # reference is computed on the host with per-op bf16 rounding instead.
    "f32_bf16_mul_f32": (lambda a: (a.astype(bf16) * bf16(3)).astype(f32), _F32),
    "f32_f16_add_f32": (lambda a: (a.astype(f16) + f16(1)).astype(f32), _F32),
    "f32_f16_bf16": (lambda a: a.astype(f16).astype(bf16), _F32),
    "i32_f16_f32": (lambda a: a.astype(f16).astype(f32), _INT),
    "i32_bf16_i32": (lambda a: a.astype(bf16).astype(jnp.int32), _INT[:-1]),
    "f16_mul_f32": (lambda a: (a * f16(2)).astype(f32), _F32.astype(np.float16)),
}


def _bits(x):
    x = np.asarray(x)
    return x.view({2: np.uint16, 4: np.uint32}[x.dtype.itemsize])


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return np.asarray(func(jnp.asarray(x)))


def _assert_same(expected, actual):
    assert actual.dtype == expected.dtype
    if np.issubdtype(expected.dtype, np.floating) or expected.dtype == jnp.bfloat16:
        e, a = expected.astype(np.float64), actual.astype(np.float64)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(e))
        keep = ~np.isnan(e)
        np.testing.assert_array_equal(_bits(actual)[keep], _bits(expected)[keep])
    else:
        np.testing.assert_array_equal(actual, expected)


_HOST_REFERENCE = {
    "f32_bf16_mul_f32": lambda a: (a.astype(bf16) * np.asarray(3, dtype=bf16)).astype(np.float32),
}


@pytest.mark.parametrize("chain", list(_CHAINS))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_cast_chain_rounds_like_cpu(chain, jit):
    fn, x = _CHAINS[chain]
    host = _HOST_REFERENCE.get(chain)
    expected = np.asarray(host(np.asarray(x))) if host else _run(fn, x, "cpu", jit)
    _assert_same(expected, _run(fn, x, "mps", jit))


@pytest.mark.parametrize("narrow", [f16, bf16], ids=["float16", "bfloat16"])
def test_scalar_cast_chain_rounds_like_cpu(narrow):
    fn = lambda a: a.astype(narrow).astype(f32)  # noqa: E731
    x = np.asarray(3.14159265, dtype=np.float32)
    _assert_same(_run(fn, x, "cpu", True), _run(fn, x, "mps", True))
