# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Registered handlers that JAX never emits must still run and match XLA:CPU bitwise.

tests/configs/util.py marks stablehlo.broadcast/dot/map and the dynamic-shape ops as exercised
without running them (they are not produced by jnp/lax). A matrix-vector stablehlo.dot aborted the
process (MPS "K dimension is inconsistently defined") while that coverage hole hid it. Each case
emits the op through a custom primitive whose lowering builds it directly, runs it under jax.jit on
CPU and MPS in a child process (so an abort fails one case), and compares the raw bytes.
Inputs are small integers, so float and f16 dots are exact on both platforms.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

CASES = [
    "broadcast_f32",
    "broadcast_i32",
    "dot_mm_f32",
    "dot_mv_f32",
    "dot_vm_f32",
    "dot_vv_f32",
    "dot_mm_i32",
    "dot_mv_i32",
    "dot_vm_i32",
    "dot_mv_f16",
    "dot_vm_f16",
    "map_f32",
    "map_i32",
    "dyn_bcast_f32",
    "dyn_reshape_f32",
    "get_dim_size",
    "set_dim_size",
]

OP_TEXT = {
    "broadcast": "stablehlo.broadcast ",
    "dot": "stablehlo.dot ",
    "map": "stablehlo.map",
    "dyn": "stablehlo.dynamic_",
    "get": "stablehlo.get_dimension_size",
    "set": "stablehlo.set_dimension_size",
}


def _run_case(name: str) -> None:
    import jax
    from jax._src import core
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import hlo
    from jax.extend.core import Primitive

    def i64(v):
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), v)

    def const_i32(vals):
        return hlo.ConstantOp(ir.DenseElementsAttr.get(np.asarray(vals, dtype=np.int32))).result

    def make(out_shape, out_dtype, lower):
        prim = Primitive(f"unemitted_{name}")
        prim.def_abstract_eval(lambda *a: core.ShapedArray(out_shape, np.dtype(out_dtype)))

        def rule(ctx, *args):
            return [lower(mlir.aval_to_ir_type(ctx.module_context, ctx.avals_out[0]), *args)]

        mlir.register_lowering(prim, rule)
        return lambda *xs: prim.bind(*xs)

    rng = np.random.default_rng(0)

    def ints(shape, dtype):
        return rng.integers(-8, 9, size=shape).astype(dtype)

    kind = name.split("_")[0]
    if kind == "broadcast":
        dt = np.float32 if name.endswith("f32") else np.int32
        args = [ints((3, 4), dt)]
        fn = make((2, 5, 3, 4), dt, lambda t, x: hlo.BroadcastOp(x, mlir.dense_int_array([2, 5])).result)
    elif kind == "dot":
        dt = {"f32": np.float32, "f16": np.float16, "i32": np.int32}[name.split("_")[2]]
        shapes = {
            "mm": ((5, 7), (7, 3), (5, 3)),
            "mv": ((5, 7), (7,), (5,)),
            "vm": ((7,), (7, 3), (3,)),
            "vv": ((7,), (7,), ()),
        }
        ls, rs, out_shape = shapes[name.split("_")[1]]
        args = [ints(ls, dt), ints(rs, dt)]
        fn = make(out_shape, dt, lambda t, a, b: hlo.DotOp(t, a, b, precision_config=None).result)
    elif kind == "map":
        dt = np.float32 if name.endswith("f32") else np.int32
        args = [ints((4, 5), dt), ints((4, 5), dt)]

        def lower_map(t, a, b):
            op = hlo.MapOp(t, [a, b], mlir.dense_int_array([0, 1]))
            st = ir.RankedTensorType.get([], ir.RankedTensorType(t).element_type)
            block = op.regions[0].blocks.append(st, st)
            with ir.InsertionPoint(block):
                p, q = block.arguments
                hlo.ReturnOp([hlo.AddOp(hlo.MulOp(p, q).result, p).result])  # a*b + a
            return op.result

        fn = make((4, 5), dt, lower_map)
    elif name == "dyn_bcast_f32":
        args = [ints((3,), np.float32)]
        fn = make(
            (2, 3),
            np.float32,
            lambda t, x: hlo.DynamicBroadcastInDimOp(t, x, const_i32([2, 3]), mlir.dense_int_array([1])).result,
        )
    elif name == "dyn_reshape_f32":
        args = [ints((3, 4), np.float32)]
        fn = make((4, 3), np.float32, lambda t, x: hlo.DynamicReshapeOp(t, x, const_i32([4, 3])).result)
    elif name == "get_dim_size":
        args = [ints((3, 4), np.float32)]

        def lower_gds(t, x):
            try:
                return hlo.GetDimensionSizeOp(x, i64(1)).result
            except TypeError:
                return hlo.GetDimensionSizeOp(t, x, i64(1)).result

        fn = make((), np.int32, lower_gds)
    elif name == "set_dim_size":
        args = [ints((3, 4), np.float32)]

        def lower_sds(t, x):
            size = hlo.ConstantOp(ir.DenseElementsAttr.get(np.asarray(4, dtype=np.int32))).result
            try:
                return hlo.SetDimensionSizeOp(x, size, i64(1)).result
            except TypeError:
                return hlo.SetDimensionSizeOp(t, x, size, i64(1)).result

        fn = make((3, 4), np.float32, lower_sds)
    else:
        raise SystemExit(f"unknown case {name}")

    outs = {}
    for plat in ("cpu", "mps"):
        dev = jax.devices(plat)[0]
        dargs = [jax.device_put(a, dev) for a in args]
        jf = jax.jit(fn)
        if plat == "mps":
            text = str(jf.lower(*dargs).compiler_ir(dialect="stablehlo"))
            assert OP_TEXT[kind] in text, f"{OP_TEXT[kind]!r} not emitted"
        out = jf(*dargs)
        assert next(iter(out.devices())).platform == plat, out.devices()
        outs[plat] = np.asarray(out)
    c, m = outs["cpu"], outs["mps"]
    assert c.shape == m.shape and c.dtype == m.dtype, (c.shape, c.dtype, m.shape, m.dtype)
    assert np.array_equal(c.reshape(-1).view(np.uint8), m.reshape(-1).view(np.uint8)), (c, m)
    print("OK")


@pytest.mark.parametrize("case", CASES)
def test_unemitted_op_matches_cpu(case):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run(
        [sys.executable, __file__, case], capture_output=True, text=True, timeout=120, env=env, check=False
    )
    assert r.returncode == 0 and "OK" in r.stdout, (r.returncode, r.stderr[-800:])


if __name__ == "__main__":
    _run_case(sys.argv[1])
