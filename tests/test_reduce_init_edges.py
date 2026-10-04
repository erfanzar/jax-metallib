# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bitwise f32 reduce-init edges, isolated in one subprocess per case.

The measured failures are an add with -0 init and all -0 data, and max/min
with a winning signed smallest-subnormal init. Compare exact-valued rows
against actual XLA:CPU bits, including mixed zeros, subnormal inputs,
cancellation and normal values that must not be changed by an edge correction.
Small, axis-0, multi-axis and native-sized reductions exercise the graph/native
boundary. No NaNs or bf16 subnormals are used: their established upstream CPU
limits are separate from these init failures.

CPU YNN REDUCE is not a valid reference for these edges: its wide max/min
passes a raw subnormal init and returns invalid extrema for
max(-smallest_subnormal, +0) / min(+smallest_subnormal, -0). It also turns
all-negative-zero sums with a genuine -0 init into +0; the same incoming IR
preserves -0 in CPU's generic loop. Neither discrepancy is reduce-tree
ambiguity. The user approved consistent -0 preservation instead of emulating
CPU compiler-dependent shape thresholds. These edge-test children disable
only CPU YNN REDUCE; other CPU flags and the parent suite stay unchanged.
This reference limitation is isolated explicitly, not skipped or xfailed.
"""

import json
import os
import shlex
import subprocess
import sys

import pytest

_CHILD = r"""
import json
import sys

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

spec = json.loads(sys.argv[1])
sub = np.array([1], np.uint32).view(np.float32)[0]
rows = np.array([
    [-0.0] * 8,
    [0.0] * 8,
    [-0.0, 0.0] * 4,
    [1.0, -1.0] * 4,
    [2.0, -1.0] * 4,
    [-2.0, 1.0] * 4,
    [sub, -sub, -0.0, 0.0] * 2,
    [-1.0, 1.0, -0.0, -0.0] * 2,
], np.float32)
if spec["op"] != "add":
    rows = np.array([
        [-2.0] * 8,
        [-0.0] * 8,
        [0.0] * 8,
        [-0.0, 0.0] * 4,
        [2.0, -1.0, -0.0, 0.0, -2.0, 1.0, -1.0, 0.0],
        [-2.0, -1.0] * 4,
        [sub, -sub, -0.0, 0.0] * 2,
        [np.finfo(np.float32).tiny, -1.0] * 4,
    ], np.float32)
    if spec["op"] == "min":
        rows = -rows

layout = spec["layout"]
if layout == "probe":
    # Match the parent's measured 10/10 failing probe exactly.
    value = {"add": -0.0, "max": -1.0, "min": 1.0}[spec["op"]]
    x, axes = np.full((4, 5), value, np.float32), (1,)
elif layout == "small":
    x, axes = rows, (1,)
elif layout == "axis0":
    x, axes = rows.T.copy(), (0,)
elif layout == "multi":
    x, axes = rows.reshape(8, 2, 4), (1, 2)
elif layout == "wide":
    # 32768 elements and a 4096-element row cross the native sum thresholds.
    x, axes = np.tile(rows, (1, 512)), (1,)
else:
    # A 4096-element full reduce is also eligible for native sum routing.
    index = {"full-negzero": 0, "full-mixedzero": 2, "full-cancel": 3}[layout]
    x, axes = np.tile(rows[index], 512), (0,)

init = np.array(spec["init_bits"], np.uint32).view(np.float32)
op = {"add": lax.add, "max": lax.max, "min": lax.min}[spec["op"]]
if spec.get("captured"):
    # Controls only: capturing a precreated JAX/NumPy -0 specializes reduce_sum
    # to +0 on CPU. This is distinct from the genuine generic -0 init below.
    constant_init = jnp.asarray(init) if spec["captured"] == "jax" else init
    fn = jax.jit(lambda a, i: lax.reduce(a, constant_init, op, axes))
else:
    def fn(a, i):
        # Keep this conversion INSIDE the traced function. Creating a JAX
        # scalar before jit incorrectly turns the original constant -0 probe
        # into a specialized +0 reduction on CPU.
        return lax.reduce(a, i if spec["traced"] else jnp.asarray(init), op, axes)

    fn = jax.jit(fn)
args = (x, init)

results = {}
for platform in ("cpu", "mps"):
    device = jax.devices(platform)[0]
    results[platform] = np.asarray(fn(*(jax.device_put(a, device) for a in args)))
want, got = results["cpu"], results["mps"]
assert want.dtype == got.dtype == np.float32, (want.dtype, got.dtype)
assert want.shape == got.shape, (want.shape, got.shape)
want_bits = want.reshape(-1).view(np.uint32)
got_bits = got.reshape(-1).view(np.uint32)
# Pin only the exact original 4x5 fn(a, i) probe with its generic init. Other
# shapes and the captured-init controls use actual CPU bits, not a fixed sign.
if layout == "probe" and not spec.get("captured"):
    expected = spec["init_bits"] & 0x80000000
    assert np.all(want_bits == expected), (
        "unexpected CPU edge oracle", spec, want_bits.tolist(), expected
    )
bad = want_bits != got_bits
print("RESULT", "BAD" if bad.any() else "OK", json.dumps(spec),
      "cpu_bits=" + str(want_bits.tolist()), "mps_bits=" + str(got_bits.tolist()), flush=True)
assert not bad.any(), (spec, "bitwise mismatch", want_bits.tolist(), got_bits.tolist())
"""

_CASES = [
    {"op": op, "init_bits": init, "traced": traced, "layout": "probe"}
    for op, init in (("add", 0x80000000), ("max", 1), ("max", 0x80000001), ("min", 1), ("min", 0x80000001))
    for traced in (False, True)
]
_CASES += [
    {"op": "add", "init_bits": init, "traced": traced, "layout": layout}
    for init in (0x80000000, 0)
    for traced in (False, True)
    for layout in ("small", "axis0", "multi", "wide", "full-negzero", "full-mixedzero", "full-cancel")
]
_CASES += [
    {"op": op, "init_bits": init, "traced": traced, "layout": layout}
    for op in ("max", "min")
    for init in (1, 0x80000001)
    for traced in (False, True)
    for layout in ("small", "axis0", "multi", "wide")
]
# Protect the already-specialized CPU +0 paths from the generic -0 fix.
_CASES += [
    {"op": "add", "init_bits": 0x80000000, "traced": False, "layout": "probe", "captured": captured}
    for captured in ("jax", "numpy")
]


def _case_id(spec):
    init = f"{spec['init_bits']:08x}"
    mode = f"captured-{spec['captured']}" if spec.get("captured") else "traced" if spec["traced"] else "constant"
    return f"{spec['op']}-init{init}-{mode}-{spec['layout']}"


def _without_cpu_ynn_reduce(flags):
    name = "--xla_cpu_experimental_ynn_fusion_type"
    tokens = shlex.split(flags)
    index = 0
    found = False
    while index < len(tokens):
        token = tokens[index]
        if token == name:
            if index + 1 == len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"missing value for {name}")
            index += 1
            value = tokens[index]
            prefix = ""
        elif token.startswith(name + "="):
            value = token[len(name) + 1 :]
            prefix = name + "="
        else:
            index += 1
            continue
        # Preserve all existing selections except REDUCE; do not replace the
        # flag wholesale or disable YNN's unrelated fusion types.
        selections = [entry for entry in value.split(",") if entry and entry.lstrip("+-") != "REDUCE"]
        tokens[index] = prefix + ",".join([*selections, "-REDUCE"])
        found = True
        index += 1
    if not found:
        tokens.append(name + "=-REDUCE")
    return shlex.join(tokens)


def _case_env(spec):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    if spec["op"] in ("add", "max", "min"):
        env["XLA_FLAGS"] = _without_cpu_ynn_reduce(env.get("XLA_FLAGS", ""))
    return env


@pytest.mark.parametrize("spec", _CASES, ids=[_case_id(spec) for spec in _CASES])
def test_reduce_init_edges_match_cpu_bits(spec):
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(spec)],
        env=_case_env(spec),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0 and "RESULT OK" in proc.stdout, (
        f"case={_case_id(spec)} rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ("", ["--xla_cpu_experimental_ynn_fusion_type=-REDUCE"]),
        (
            "--xla_allow_excess_precision=false",
            ["--xla_allow_excess_precision=false", "--xla_cpu_experimental_ynn_fusion_type=-REDUCE"],
        ),
        (
            "--xla_cpu_experimental_ynn_fusion_type=-REDUCE",
            ["--xla_cpu_experimental_ynn_fusion_type=-REDUCE"],
        ),
        (
            "--xla_cpu_experimental_ynn_fusion_type=REDUCE,-DOT,TRANSPOSE --xla_allow_excess_precision=false",
            ["--xla_cpu_experimental_ynn_fusion_type=-DOT,TRANSPOSE,-REDUCE", "--xla_allow_excess_precision=false"],
        ),
        (
            "--xla_cpu_experimental_ynn_fusion_type=+REDUCE,+DOT",
            ["--xla_cpu_experimental_ynn_fusion_type=+DOT,-REDUCE"],
        ),
        (
            "--xla_cpu_experimental_ynn_fusion_type REDUCE,-DOT",
            ["--xla_cpu_experimental_ynn_fusion_type", "-DOT,-REDUCE"],
        ),
        (
            "--xla_cpu_experimental_ynn_fusion_type=REDUCE --xla_cpu_experimental_ynn_fusion_type=-REDUCE,TRANSPOSE",
            [
                "--xla_cpu_experimental_ynn_fusion_type=-REDUCE",
                "--xla_cpu_experimental_ynn_fusion_type=TRANSPOSE,-REDUCE",
            ],
        ),
    ],
)
def test_cpu_reference_flags_preserve_other_selections(flags, expected):
    assert shlex.split(_without_cpu_ynn_reduce(flags)) == expected


@pytest.mark.parametrize("flags", [None, "--xla_cpu_experimental_ynn_fusion_type=REDUCE,-DOT"])
def test_unrelated_reference_environment_unchanged(monkeypatch, flags):
    if flags is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", flags)
    original = os.environ.copy()
    assert _case_env({"op": "mul"}) == dict(original, JAX_PLATFORMS="cpu,mps")
    assert dict(os.environ) == original


@pytest.mark.parametrize("op", ["add", "max", "min"])
def test_cpu_reference_flags_are_child_local(monkeypatch, op):
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_cpu_experimental_ynn_fusion_type=REDUCE,-DOT --xla_allow_excess_precision=false"
    )
    original = os.environ.copy()
    env = _case_env({"op": op})
    assert shlex.split(env["XLA_FLAGS"]) == [
        "--xla_cpu_experimental_ynn_fusion_type=-DOT,-REDUCE",
        "--xla_allow_excess_precision=false",
    ]
    assert env == dict(original, JAX_PLATFORMS="cpu,mps", XLA_FLAGS=env["XLA_FLAGS"])
    assert dict(os.environ) == original
