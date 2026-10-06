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

"""JAX primitive and static launch descriptor for native custom Metal kernels."""

from __future__ import annotations

import json
import operator
import re
import struct
from dataclasses import dataclass
from functools import lru_cache

import jax
import numpy as np
from jax._src import core, dispatch
from jax._src.interpreters import ad, batching, mlir

_DTYPES = frozenset(
    np.dtype(name) for name in ("float32", "float16", "int32", "uint32", "int16", "uint16", "int8", "uint8", "bool")
)
_TARGET = "jax_metallib.metal_kernel"
_primitive = core.Primitive("native_metal_kernel")
_primitive.multiple_results = True
_primitive.def_impl(lambda *args, **params: dispatch.apply_primitive(_primitive, *args, **params))


def _abstract_eval(*inputs, outputs, backend_config):
    del inputs, backend_config
    return [core.ShapedArray(shape, np.dtype(dtype)) for shape, dtype in outputs]


_primitive.def_abstract_eval(_abstract_eval)


def _lower(ctx, *operands, outputs, backend_config):
    del outputs
    if set(ctx.module_context.platforms) != {"mps"}:
        _unsupported_platform()
    return mlir.custom_call(
        _TARGET,
        result_types=[mlir.aval_to_ir_type(ctx.module_context, aval) for aval in ctx.avals_out],
        operands=operands,
        backend_config=backend_config,
        api_version=1,
        operand_layouts=[tuple(reversed(range(aval.ndim))) for aval in ctx.avals_in],
        result_layouts=[tuple(reversed(range(aval.ndim))) for aval in ctx.avals_out],
    ).results


def _unsupported_platform(*args, **kwargs):
    raise NotImplementedError(
        "Native Metal kernels require JAX's mps backend; no CPU or other-device fallback is provided"
    )


def _unsupported_jvp(*args, **kwargs):
    raise NotImplementedError("Native Metal kernel differentiation requires a user-supplied custom JVP or VJP rule")


def _unsupported_batching(*args, **kwargs):
    raise NotImplementedError("Native Metal kernels do not yet support vmap; supply an explicit batching rule")


# Plugin platforms are discovered lazily, after this module may be imported.
# A common rule with an explicit platform check avoids eager GPU initialization.
mlir.register_lowering(_primitive, _lower)
ad.primitive_jvps[_primitive] = _unsupported_jvp
batching.primitive_batchers[_primitive] = _unsupported_batching


def _dimensions(value, *, name, allow_zero):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        values = (value, 1, 1)
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        values = value
    else:
        raise TypeError(f"{name} must be an integer or a three-integer tuple")
    result = []
    for dimension in values:
        if isinstance(dimension, (bool, np.bool_)):
            raise TypeError(f"{name} dimensions must be integers, not bool")
        try:
            dimension = operator.index(dimension)
        except TypeError as error:
            raise TypeError(f"{name} dimensions must be integers") from error
        if dimension < (0 if allow_zero else 1) or dimension > 2**32 - 1:
            raise ValueError(f"{name} dimensions must be {'nonnegative' if allow_zero else 'positive'} uint32 values")
        result.append(dimension)
    return tuple(result)


def _dtype(value):
    dtype = np.dtype(value)
    if dtype not in _DTYPES:
        raise TypeError(f"Unsupported native Metal kernel dtype: {dtype}")
    return dtype


def _outputs(out):
    single = isinstance(out, jax.ShapeDtypeStruct)
    specs = (out,) if single else tuple(out) if isinstance(out, (tuple, list)) else ()
    if not specs or not all(isinstance(spec, jax.ShapeDtypeStruct) for spec in specs):
        raise ValueError("out must be a ShapeDtypeStruct or a nonempty sequence of ShapeDtypeStruct outputs")
    results = []
    for spec in specs:
        dtype = _dtype(spec.dtype)
        shape = []
        count = 1
        for dim in spec.shape:
            if isinstance(dim, (bool, np.bool_)):
                raise TypeError("out shape dimensions must be static nonnegative integers")
            try:
                dim = operator.index(dim)
            except TypeError as error:
                raise TypeError("out shape dimensions must be static nonnegative integers") from error
            if not 0 <= dim <= 2**63 - 1:
                raise ValueError("out shape dimensions must be nonnegative and fit int64")
            shape.append(dim)
            count *= dim
        if count > (2**63 - 1) // dtype.itemsize:
            raise ValueError("out buffer byte size overflows int64")
        results.append((tuple(shape), dtype.str))
    return tuple(results), single


def _constants(constants, buffer_count):
    if constants is None:
        return []
    if not isinstance(constants, dict):
        raise TypeError("constants must map Metal buffer indices to static typed values or bytes")
    packed = []
    for index, value in constants.items():
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("constant buffer indices must be integers, not bool")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise TypeError("constant buffer indices must be integers") from error
        if not buffer_count <= index < 31:
            raise ValueError("constant buffer indices must not overlap array buffers and must be below 31")
        if isinstance(value, core.Tracer) or isinstance(value, jax.Array):
            raise TypeError("constants must be static host values; pass dynamic JAX values as input buffers")
        if isinstance(value, bytes):
            data = value
        elif isinstance(value, (np.generic, np.ndarray)):
            dtype = _dtype(value.dtype)
            data = np.asarray(value, dtype=dtype.newbyteorder("<")).tobytes(order="C")
        elif isinstance(value, bool):
            data = struct.pack("<?", value)
        elif isinstance(value, int):
            if not 0 <= value <= 2**32 - 1:
                raise ValueError(
                    "Python integer constants must fit uint32; use an explicit NumPy integer dtype otherwise"
                )
            data = struct.pack("<I", value)
        elif isinstance(value, float):
            data = struct.pack("<f", value)
        else:
            raise TypeError("constant values must be static typed NumPy values, Python scalars, or bytes")
        if not 0 < len(data) <= 4096:
            raise ValueError("inline constant payloads must contain 1 to 4096 bytes")
        packed.append({"index": index, "hex": data.hex()})
    return sorted(packed, key=lambda item: item["index"])


@lru_cache(maxsize=128)
def _prepare_launch(source, entry_point, outputs, threads, threadgroup, constants):
    # Cache only validated immutable metadata, never JAX arrays or results.
    descriptor = {
        "version": 1,
        "source": source,
        "entry_point": entry_point,
        "threads": threads,
        "threadgroup": threadgroup,
        "constants": [{"index": index, "hex": data} for index, data in constants],
    }
    config = json.dumps(descriptor, separators=(",", ":"), ensure_ascii=True)

    @jax.jit(inline=True)
    def execute(*inputs):
        return _primitive.bind(*inputs, outputs=outputs, backend_config=config)

    return config, execute


@dataclass(frozen=True)
class CompiledMetalKernel:
    """Immutable lazy native kernel definition, executable eagerly or within jax.jit.

    Inputs are read-only, contiguous row-major buffers. Outputs follow inputs in
    Metal's buffer bindings. The kernel must initialize every output element and
    must not write inputs. Launch geometry and inline constants are static JAX
    compilation parameters. Native pipeline compilation happens in the backend.
    """

    source: str
    entry_point: str

    def __call__(self, *inputs, out, threads, threadgroup=256, constants=None):
        outputs, single = _outputs(out)
        if len(inputs) + len(outputs) > 31:
            raise ValueError("native Metal kernels support at most 31 array buffer bindings")
        for value in inputs:
            if not isinstance(value, (jax.Array, core.Tracer)):
                raise TypeError("native Metal inputs must be JAX arrays; use jax.device_put to place host data")
            _dtype(value.dtype)
        threads = _dimensions(threads, name="threads", allow_zero=True)
        threadgroup = _dimensions(threadgroup, name="threadgroup", allow_zero=False)
        packed = _constants(constants, len(inputs) + len(outputs))
        config, execute = _prepare_launch(
            self.source,
            self.entry_point,
            outputs,
            threads,
            threadgroup,
            tuple((item["index"], item["hex"]) for item in packed),
        )
        if inputs and core.trace_state_clean():
            result = execute(*inputs)
        else:
            # Keep staging and zero-input device selection on the original path.
            result = _primitive.bind(*inputs, outputs=outputs, backend_config=config)
        return result[0] if single else tuple(result)


def compile_kernel(source: str, *, entry_point: str) -> CompiledMetalKernel:
    """Define a native Metal kernel; source is compiled lazily by the MPS backend.

    Unlike metal.execute, this path has no PyObjC or host-array-copy dependency.
    It works with eager MPS arrays and inside jax.jit. Autodiff, vmap and nested
    control-flow execution need explicit rules or future backend support.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a nonempty Metal Shading Language string")
    if not isinstance(entry_point, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_point):
        raise ValueError("entry_point must be a valid Metal function identifier")
    return CompiledMetalKernel(source, entry_point)
