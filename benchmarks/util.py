from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import jax

# Matches StableHLO and CHLO op names in textual IR.
# Keep in sync with `tests/configs/util.py` but duplicated here to avoid importing from tests.
STABLEHLO_OP_RE = re.compile(r"(?<![\#\!])(?:stablehlo|chlo)\.[\w\.]+")


def now_s() -> float:
    return time.perf_counter()


def block_until_ready_pytree(x: Any) -> None:
    def _block(leaf):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
        return leaf

    jax.tree_util.tree_map(_block, x)


def stablehlo_text(lowered) -> str:
    return str(lowered.compiler_ir(dialect="stablehlo"))


def stablehlo_op_counts_from_text(text: str) -> dict[str, int]:
    return dict(Counter(STABLEHLO_OP_RE.findall(text)))


def summarize_pytree_shapes_and_dtypes(x: Any) -> tuple[list[str], list[str]]:
    shapes: list[str] = []
    dtypes: list[str] = []
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            shapes.append(str(tuple(int(d) for d in leaf.shape)))
            dtypes.append(str(leaf.dtype))
        else:
            shapes.append(type(leaf).__name__)
            dtypes.append(type(leaf).__name__)
    return shapes, dtypes


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


@dataclass(frozen=True)
class Timing:
    lower_s: float | None
    compile_s: float | None
    first_run_s: float
    steady_state_s: float
