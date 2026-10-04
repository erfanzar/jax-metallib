"""Non-identity reduce init values vs XLA:CPU, bitwise, one child process per case.

MPSGraph reductions take no init, and HandleSingleResultReduce dropped a non-identity
`init` for add/mul/max/min: `lax.reduce(x, 5., lax.add, (1,))` returned the plain sum, a
max init that beats the data was ignored, and a NaN init did not propagate
(build/probe_r20k_reduce_init*.log). The init is now combined after the reduction.

Data is chosen so every result is exact (small integers for add, +-{0.5, 1, 2} for mul),
so float add/mul are compared bitwise too and summation order cannot mask a dropped init.
Covers constant and traced inits, float/int/unsigned dtypes, single/multi/full axes, shapes
that reach the planner's native reduce bypass, NaN inits and -0 < +0 ties with a zero init.
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
import ml_dtypes
import numpy as np
from jax import lax

spec = json.loads(sys.argv[1])
try:
    mps = jax.devices("mps")[0]
except RuntimeError:
    print("RESULT SKIP no mps device", flush=True)
    sys.exit(0)
cpu = jax.devices("cpu")[0]
dt = np.dtype(ml_dtypes.bfloat16) if spec["dtype"] == "bfloat16" else np.dtype(spec["dtype"])
shape = tuple(spec["shape"])
op_name = spec["op"]
rng = np.random.default_rng(spec["seed"])
data = spec["data"]
if data == "negzero":
    x = np.full(shape, -0.0)
elif data == "poszero":
    x = np.zeros(shape)
elif dt == np.bool_:
    x = rng.random(shape) < 0.5
elif op_name == "mul":
    x = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], shape) if dt.kind == "f" else rng.integers(-2, 3, shape)
elif dt.kind == "u":
    x = rng.integers(0, 50, shape)
else:
    lim = 4 if spec["dtype"] == "bfloat16" else 50 if dt.kind == "i" else 8
    x = rng.integers(-lim, lim + 1, shape)
x = np.asarray(x).astype(dt)
init = np.asarray(float(spec["init"]) if dt.kind == "f" or spec["dtype"] == "bfloat16" else spec["init"]).astype(dt)
op = {"add": lax.add, "mul": lax.mul, "max": lax.max, "min": lax.min}[op_name]
axes = tuple(spec["axes"])

if spec["traced"]:
    fn = jax.jit(lambda a, i: lax.reduce(a, i, op, axes))
    run = lambda dev: np.asarray(fn(jax.device_put(x, dev), jax.device_put(init, dev)))
else:
    fn = jax.jit(lambda a: lax.reduce(a, init[()], op, axes))
    run = lambda dev: np.asarray(fn(jax.device_put(x, dev)))
ref = run(cpu)
got = run(mps)
same_shape = ref.shape == got.shape and ref.dtype == got.dtype
if same_shape and ref.dtype != np.bool_:
    ubits = f"uint{ref.dtype.itemsize * 8}"
    mism = int((ref.view(ubits) != got.view(ubits)).sum())
else:
    mism = int((ref != got).sum()) if same_shape else -1
ok = same_shape and mism == 0
head = lambda a: a.ravel()[:3].astype(np.float64).tolist()
print(f"RESULT {'OK' if ok else 'BAD'} shape={got.shape}/{ref.shape} dtype={got.dtype}/{ref.dtype} "
      f"mism={mism} mps={head(got)} cpu={head(ref)}")
"""

CASES = []


def _add(op, dtype, shape, axes, init, traced=False, data="rand"):
    CASES.append(
        {
            "op": op,
            "dtype": dtype,
            "shape": shape,
            "axes": axes,
            "init": init,
            "traced": traced,
            "data": data,
            "seed": len(CASES),
        }
    )


# add / mul: constant and traced, every float dtype plus integers.
for _dt in ("float32", "float16", "bfloat16", "int32"):
    _add("add", _dt, [6, 9], [1], 5)
    _add("mul", _dt, [5, 4], [0], 3 if _dt == "int32" else 2)
_add("add", "float32", [6, 9], [0, 1], 5)
_add("add", "float32", [6, 9], [0], 2.5, traced=True)
_add("add", "int8", [6, 9], [1], -7)
_add("add", "uint16", [6, 9], [1], 300)
_add("mul", "float32", [4, 6], [1], -0.5, traced=True)
# max / min: an init that beats every row, beats some rows, is traced, is NaN.
for _dt in ("float32", "float16", "bfloat16", "int32"):
    _add("max", _dt, [6, 9], [1], 100)
    _add("min", _dt, [6, 9], [0], -100)
_add("max", "float32", [6, 9], [1], 6.5)
_add("min", "int32", [6, 9], [1], -40)
_add("max", "float32", [6, 9], [0], 100, traced=True)
_add("min", "float32", [6, 9], [1], -3, traced=True)
_add("max", "float32", [6, 9], [1], "nan")
_add("min", "float16", [6, 9], [0, 1], "nan")
_add("max", "uint8", [6, 9], [1], 200)
_add("min", "uint8", [6, 9], [1], 3)
_add("max", "int8", [6, 9], [0], 100)
# -0 < +0 ties between a zero init and an all-zero row.
_add("max", "float32", [4, 5], [1], 0.0, data="negzero")
_add("max", "float32", [4, 5], [1], 0.0, traced=True, data="negzero")
_add("min", "float32", [4, 5], [1], -0.0, data="poszero")
_add("min", "float32", [4, 5], [1], -0.0, traced=True, data="poszero")
_add("max", "float32", [4, 5], [1], -0.0, data="negzero")
_add("add", "float32", [4, 5], [1], 0.0, data="negzero")
# Shapes that reach the native reduce bypass for neutral inits (rows >= 1024, full).
_add("add", "float32", [4, 2048], [1], 5)
_add("add", "float32", [4096], [0], 5)
_add("add", "float32", [64, 64], [0, 1], 3, traced=True)
_add("max", "float32", [4096], [0], 1000)
_add("min", "float32", [4096], [0], -1000)
_add("max", "float32", [64, 64], [0, 1], "nan")
# Identity-init controls keep the existing lowering.
_add("add", "float32", [4, 2048], [1], 0.0)
_add("max", "float32", [4096], [0], "-inf")
_add("mul", "int32", [5, 4], [0], 1)
_add("min", "uint8", [6, 9], [1], 255)


def _case_id(spec):
    shape = "x".join(map(str, spec["shape"]))
    axes = "".join(map(str, spec["axes"]))
    extra = ("-traced" if spec["traced"] else "") + ("" if spec["data"] == "rand" else f"-{spec['data']}")
    return f"{spec['op']}-{spec['dtype']}-{shape}-ax{axes}-init{spec['init']}{extra}"


@pytest.mark.parametrize("spec", CASES, ids=[_case_id(c) for c in CASES])
def test_reduce_init(spec):
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
