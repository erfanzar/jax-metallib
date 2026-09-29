# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Concatenate with zero-size operands must match XLA CPU on MPS.

Under jit, MPSGraph concatTensors with a zero-width operand in a trailing or
middle position shifted the result (e.g. rows came back offset by one element),
which broke jnp.linalg.svd(full_matrices=True) for square inputs. Zero-size
operands along the concat axis are now dropped before concatenation.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_X = (np.arange(25, dtype=np.float32).reshape(5, 5) + 1).astype(np.float32)

_CASES = {
    "empty_tail_axis1": lambda a: jnp.concatenate([a, a[:, 5:]], axis=1),
    "empty_head_axis1": lambda a: jnp.concatenate([a[:, :0], a], axis=1),
    "empty_mid_axis0": lambda a: jnp.concatenate([a[:2], a[5:], a[2:]], axis=0),
    "scaled_empty_tail": lambda a: jnp.concatenate([a * 2, (a * 3)[:, 5:]], axis=1),
    "two_empty": lambda a: jnp.concatenate([a[:, :0], a * 2, a[:, 5:]], axis=1),
    "all_empty": lambda a: jnp.concatenate([a[:, :0], a[:, 5:]], axis=1),
    "int32_empty_tail": lambda a: jnp.concatenate([a.astype(jnp.int32), a[:, 5:].astype(jnp.int32)], axis=1),
    "nonempty_control": lambda a: jnp.concatenate([a[:, :3], a[:, 3:]], axis=1),
}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("case", list(_CASES))
def test_concat_with_empty_operand_matches_cpu(case, jit):
    fn = jax.jit(_CASES[case]) if jit else _CASES[case]
    out = {}
    for p in ("cpu", "mps"):
        with jax.default_device(jax.devices(p)[0]):
            out[p] = np.asarray(fn(jnp.asarray(_X)))
    assert out["mps"].dtype == out["cpu"].dtype and out["mps"].shape == out["cpu"].shape
    np.testing.assert_array_equal(out["mps"], out["cpu"])
