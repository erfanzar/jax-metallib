# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MPS lowering for ``lax.sort_p`` that moves a non-last sort dimension to the end.

The execution plan only routes a ``stablehlo.sort`` to the native bitonic kernel
when it sorts the last axis; other axes ran the MPSGraph argsort + gather path.
f32 ``lax.sort(x, dimension=0)``: [256,4096] 8.34 -> 0.79 ms, [64,65536] 87.9 ->
2.06 ms, [16,4096] 4.94 -> 0.29 ms; no measured case got slower, and results are
bit-exact against XLA:CPU. ``JAX_METALLIB_DISABLE_SORT_MOVE_AXIS=1`` keeps JAX's
default lowering.
"""

from __future__ import annotations

import os


def _disabled() -> bool:
    return os.environ.get("JAX_METALLIB_DISABLE_SORT_MOVE_AXIS", "0") not in ("", "0")


def register() -> None:
    """Register the MPS sort lowering. Call after the 'mps' plugin is registered."""
    if _disabled():
        return
    from jax import lax
    from jax._src.interpreters import mlir
    from jax._src.lax import lax as lax_internal

    def _moved_sort(*operands, dimension, is_stable, num_keys):
        last = operands[0].ndim - 1
        perm = [d for d in range(last + 1) if d != dimension] + [dimension]
        inv = [0] * (last + 1)
        for i, d in enumerate(perm):
            inv[d] = i
        moved = [lax.transpose(x, perm) for x in operands]
        out = lax_internal.sort_p.bind(*moved, dimension=last, is_stable=is_stable, num_keys=num_keys)
        return [lax.transpose(x, inv) for x in out]

    moved_rule = mlir.lower_fun(_moved_sort, multiple_results=True)

    def _sort_rule(ctx, *operands, dimension, is_stable, num_keys):
        shape = ctx.avals_in[0].shape
        if len(shape) >= 2 and dimension != len(shape) - 1 and shape[dimension] > 1:
            return moved_rule(ctx, *operands, dimension=dimension, is_stable=is_stable, num_keys=num_keys)
        return lax_internal._sort_lower(ctx, *operands, dimension=dimension, is_stable=is_stable, num_keys=num_keys)

    mlir.register_lowering(lax_internal.sort_p, _sort_rule, platform="mps")
