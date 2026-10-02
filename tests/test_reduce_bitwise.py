"""Bitwise reductions (xor / and / or) vs XLA:CPU, bitwise, one child process per case.

`lax.reduce(x, 0, lax.bitwise_xor, axes)` failed to compile on MPS with "reduce: unsupported
reduction type: stablehlo.xor" (MPSGraph has no xor reduction). Integer and/or reductions are
checked alongside, since MPSGraph's reductionAnd/reductionOr are not documented as bitwise.
Covers every 8/16/32-bit integer dtype and bool, single/multi/full axes, odd lengths, length 1,
non-identity init, a 2^20 full reduce, a long last axis, and empty operands.
"""

import json
import os
import subprocess
import sys

import pytest

CHILD = r"""
import json
import sys

import jax
import numpy as np
from jax import lax

spec = json.loads(sys.argv[1])
try:
    mps = jax.devices("mps")[0]
except RuntimeError:
    print("RESULT SKIP no mps device", flush=True)
    sys.exit(0)
cpu = jax.devices("cpu")[0]
dt = np.dtype(spec["dtype"])
shape = tuple(spec["shape"])
op_name = spec["op"]
rng = np.random.default_rng(spec["seed"])
if dt == np.bool_:
    u = rng.random(shape)
    x = u > 0.1 if op_name == "and" else (u < 0.1 if op_name == "or" else u < 0.5)
else:
    bits = dt.itemsize * 8
    if op_name == "xor":
        info = np.iinfo(dt)
        x = rng.integers(info.min, int(info.max) + 1, shape, dtype=np.int64).astype(dt)
    else:
        one_bit = np.left_shift(np.ones(shape, dt), rng.integers(0, bits, shape).astype(dt))
        x = ~one_bit if op_name == "and" else one_bit
if spec["init"] == "id":
    init = ~np.zeros((), dt) if op_name == "and" else np.zeros((), dt)
else:
    init = np.asarray(spec["init"]).astype(dt)
init = init[()]
op = {"xor": lax.bitwise_xor, "and": lax.bitwise_and, "or": lax.bitwise_or}[op_name]
axes = tuple(spec["axes"])


def fn(a):
    return lax.reduce(a, init, op, axes)


ref = np.asarray(jax.jit(fn)(jax.device_put(x, cpu)))
got = np.asarray(jax.jit(fn)(jax.device_put(x, mps)))
same_shape = ref.shape == got.shape
ok = same_shape and ref.dtype == got.dtype and np.array_equal(ref, got)
mism = int((ref != got).sum()) if same_shape else -1
print(f"RESULT {'OK' if ok else 'BAD'} shape={got.shape}/{ref.shape} dtype={got.dtype}/{ref.dtype} mism={mism}")
"""

INTS = ["uint8", "int8", "uint16", "int16", "uint32", "int32"]
CASES = []


def _add(op, dtype, shape, axes, init="id"):
    CASES.append({"op": op, "dtype": dtype, "shape": shape, "axes": axes, "init": init, "seed": len(CASES)})


for _op in ("xor", "and", "or"):
    for _dt in INTS:
        _add(_op, _dt, [5, 7, 3], [1])
    for _axes in ([0], [0, 2], [0, 1, 2]):
        _add(_op, "uint32", [5, 7, 3], _axes)
    _add(_op, "int16", [33], [0])
    _add(_op, "int8", [1], [0])
    _add(_op, "bool", [5, 7, 3], [1])
    _add(_op, "bool", [5, 7, 3], [0, 1, 2])
    _add(_op, "uint32", [1 << 20], [0])
    _add(_op, "int32", [64, 4097], [1])
_add("xor", "uint16", [6, 9], [1], init=0x5A5A)
_add("and", "uint8", [6, 9], [0], init=0xF0)
_add("or", "int32", [6, 9], [1], init=0x1234)
_add("xor", "uint32", [0, 5], [1])
_add("xor", "uint32", [5, 0], [1])


def _case_id(spec):
    shape = "x".join(map(str, spec["shape"]))
    axes = "".join(map(str, spec["axes"]))
    return f"{spec['op']}-{spec['dtype']}-{shape}-ax{axes}-init{spec['init']}"


@pytest.mark.parametrize("spec", CASES, ids=[_case_id(c) for c in CASES])
def test_reduce_bitwise(spec):
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=os.environ.copy(),
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), "")
    if line.startswith("RESULT SKIP"):
        pytest.skip(line[12:])
    assert proc.returncode == 0 and line, f"rc={proc.returncode} {proc.stderr.strip()[-1500:]}"
    assert line.startswith("RESULT OK"), line
