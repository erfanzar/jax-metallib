# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""lax.axis_index under shard_map / pmap must work on the single MPS device.

Both lower to stablehlo.partition_id, which had no MPS handler. With one device
the partition id is always 0.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def _mps():
    return jax.devices("mps")[0]


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_shard_map_axis_index_is_zero(jit):
    dev = _mps()
    mesh = Mesh(np.array([dev]), ("x",))

    def body(a):
        return a + lax.axis_index("x").astype(a.dtype) * 10

    fn = jax.shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False)
    if jit:
        fn = jax.jit(fn)
    x = np.arange(4.0, dtype=np.float32)
    with jax.default_device(dev):
        out = np.asarray(fn(jnp.asarray(x)))
    np.testing.assert_array_equal(out, x)


def test_pmap_axis_index_is_zero():
    dev = _mps()
    fn = jax.pmap(lambda a: a + lax.axis_index("i").astype(a.dtype) * 10, axis_name="i", devices=[dev])
    x = np.arange(4.0, dtype=np.float32).reshape(1, 4)
    np.testing.assert_array_equal(np.asarray(fn(x)), x)


def test_axis_index_dtype_and_value():
    dev = _mps()
    mesh = Mesh(np.array([dev]), ("x",))

    def body(a):
        return jnp.broadcast_to(lax.axis_index("x"), a.shape)

    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False))
    with jax.default_device(dev):
        out = np.asarray(fn(jnp.zeros((3,), jnp.float32)))
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, np.zeros((3,), np.int32))
