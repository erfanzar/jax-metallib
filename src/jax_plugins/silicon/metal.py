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

"""Custom Metal kernel API for jax-metallib.

Write Metal compute kernels and execute them on JAX arrays directly on GPU.

Usage:
    from jax_plugins.silicon import metal

    # Simple functional API
    source = '''
    #include <metal_stdlib>
    using namespace metal;
    kernel void my_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
        c[gid] = a[gid] + b[gid];
    }
    '''
    result = metal.execute(source, "my_add",
                           inputs=[a, b],
                           output_shapes=[(1024,)],
                           output_dtypes=[jnp.float32],
                           grid=(1024, 1, 1),
                           threadgroup=(256, 1, 1))

    # Object-oriented API with caching
    kernel = metal.MetalKernel(source, "my_add")
    result = kernel(inputs=[a, b],
                    output_shapes=[(1024,)],
                    output_dtypes=[jnp.float32],
                    grid=(1024, 1, 1),
                    threadgroup=(256, 1, 1))

    # Decorator API
    @metal.kernel("my_relu", grid=lambda n: (n, 1, 1), threadgroup=(256, 1, 1))
    def relu_source():
        return '''
        kernel void my_relu(device const float* x [[buffer(0)]],
                            device float* y [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
            y[gid] = max(0.0f, x[gid]);
        }
        '''
    result = relu_source(x, output_shapes=[(1024,)], output_dtypes=[jnp.float32])
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

try:
    import Metal

    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False

# ──────────────────────────────────────────────────────────────────
# Metal Device + Pipeline Cache
# ──────────────────────────────────────────────────────────────────

_device = None
_command_queue = None
_library_cache: dict[str, object] = {}
_pipeline_cache: dict[str, object] = {}


def _ensure_metal():
    """Lazily initialize Metal device and command queue."""
    global _device, _command_queue
    if not _HAS_METAL:
        raise RuntimeError("PyObjC Metal bindings not found. Install with:\n  pip install pyobjc-framework-Metal")
    if _device is None:
        _device = Metal.MTLCreateSystemDefaultDevice()
        if _device is None:
            raise RuntimeError("No Metal device available")
        _command_queue = _device.newCommandQueue()
    return _device, _command_queue


def _get_pipeline(source: str, function_name: str):
    """Compile Metal source and return a pipeline state (cached)."""
    device, _ = _ensure_metal()

    # Cache key is hash of source + function name
    key = hashlib.sha256(f"{source}::{function_name}".encode()).hexdigest()

    if key in _pipeline_cache:
        return _pipeline_cache[key]

    # Compile library (cached by source hash)
    src_hash = hashlib.sha256(source.encode()).hexdigest()
    if src_hash not in _library_cache:
        lib, err = device.newLibraryWithSource_options_error_(source, None, None)
        if lib is None:
            err_str = str(err) if err else "unknown error"
            raise RuntimeError(f"Metal compilation failed:\n{err_str}")
        _library_cache[src_hash] = lib
    else:
        lib = _library_cache[src_hash]

    # Get function
    fn = lib.newFunctionWithName_(function_name)
    if fn is None:
        names = list(lib.functionNames())
        raise ValueError(f"Kernel function '{function_name}' not found. Available: {names}")

    # Create pipeline
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        err_str = str(err) if err else "unknown error"
        raise RuntimeError(f"Pipeline creation failed for '{function_name}': {err_str}")

    _pipeline_cache[key] = pipeline
    return pipeline


# ──────────────────────────────────────────────────────────────────
# JAX Array ↔ Metal Buffer Bridge
# ──────────────────────────────────────────────────────────────────

_DTYPE_TO_BYTES = {
    jnp.float32: 4,
    jnp.float16: 2,
    jnp.bfloat16: 2,
    jnp.int32: 4,
    jnp.int16: 2,
    jnp.int8: 1,
    jnp.uint32: 4,
    jnp.uint16: 2,
    jnp.uint8: 1,
    jnp.bool_: 1,
}


def _jax_to_numpy(x):
    """Convert JAX array to numpy, handling device transfer."""
    return np.asarray(x)


def _create_metal_buffer(data: np.ndarray):
    """Create a Metal shared buffer from numpy array."""
    device, _ = _ensure_metal()
    byte_data = data.tobytes()
    buf = device.newBufferWithBytes_length_options_(byte_data, len(byte_data), Metal.MTLResourceStorageModeShared)
    return buf


def _create_output_buffer(shape, dtype):
    """Create an empty Metal shared buffer for output."""
    device, _ = _ensure_metal()
    nbytes = int(np.prod(shape)) * _DTYPE_TO_BYTES.get(dtype, np.dtype(dtype).itemsize)
    buf = device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)
    return buf, nbytes


# ──────────────────────────────────────────────────────────────────
# Constant Buffer Support
# ──────────────────────────────────────────────────────────────────


def _pack_constants(constants: dict[int, object] | None) -> dict[int, bytes]:
    """Pack constant values into bytes for Metal setBytes."""
    if not constants:
        return {}
    result = {}
    for idx, val in constants.items():
        if isinstance(val, int):
            result[idx] = struct.pack("<I", val)  # uint32
        elif isinstance(val, float):
            result[idx] = struct.pack("<f", val)  # float32
        elif isinstance(val, bytes):
            result[idx] = val
        elif isinstance(val, np.ndarray):
            result[idx] = val.tobytes()
        elif isinstance(val, (tuple, list)):
            # Pack as array of uint32
            result[idx] = struct.pack(f"<{len(val)}I", *[int(v) for v in val])
        else:
            raise TypeError(f"Unsupported constant type at index {idx}: {type(val)}")
    return result


# ──────────────────────────────────────────────────────────────────
# Core Execution
# ──────────────────────────────────────────────────────────────────


def execute(
    source: str,
    function_name: str,
    *,
    inputs: Sequence[jax.Array] = (),
    output_shapes: Sequence[tuple[int, ...]] = (),
    output_dtypes: Sequence = (),
    grid: tuple[int, int, int] = (1, 1, 1),
    threadgroup: tuple[int, int, int] = (1, 1, 1),
    constants: dict[int, object] | None = None,
) -> list[jax.Array]:
    """Execute a custom Metal kernel on JAX arrays.

    Args:
        source: Metal kernel source code (MSL). Must include #include <metal_stdlib>.
        function_name: Name of the kernel function to execute.
        inputs: Input JAX arrays. Bound to buffer(0), buffer(1), etc.
        output_shapes: Shape of each output array.
        output_dtypes: Dtype of each output array.
        grid: Threadgroup grid dimensions (width, height, depth).
        threadgroup: Threads per threadgroup (width, height, depth).
        constants: Dict mapping buffer index → constant value.
            Values are packed as uint32 (int), float32 (float), or raw bytes.
            These use setBytes instead of setBuffer for small values.

    Returns:
        List of output JAX arrays.

    Example:
        >>> source = '''
        ... #include <metal_stdlib>
        ... using namespace metal;
        ... kernel void scale(device const float* in [[buffer(0)]],
        ...                   device float* out [[buffer(1)]],
        ...                   constant float& factor [[buffer(2)]],
        ...                   uint gid [[thread_position_in_grid]]) {
        ...     out[gid] = in[gid] * factor;
        ... }
        ... '''
        >>> out = metal.execute(source, "scale",
        ...     inputs=[x], output_shapes=[(1024,)], output_dtypes=[jnp.float32],
        ...     constants={2: 3.14},
        ...     grid=(1024, 1, 1), threadgroup=(256, 1, 1))
    """
    if len(output_shapes) != len(output_dtypes):
        raise ValueError("output_shapes and output_dtypes must have same length")

    pipeline = _get_pipeline(source, function_name)
    _, command_queue = _ensure_metal()

    # Prepare input buffers
    input_bufs = []
    for x in inputs:
        np_data = np.ascontiguousarray(_jax_to_numpy(x))
        buf = _create_metal_buffer(np_data)
        input_bufs.append(buf)

    # Prepare output buffers
    output_bufs = []
    output_nbytes = []
    for shape, dtype in zip(output_shapes, output_dtypes, strict=False):
        buf, nbytes = _create_output_buffer(shape, dtype)
        output_bufs.append(buf)
        output_nbytes.append(nbytes)

    # Pack constants
    packed_constants = _pack_constants(constants)

    # Create command buffer and encoder
    cmd_buf = command_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()

    # Set pipeline
    encoder.setComputePipelineState_(pipeline)

    # Bind input buffers
    buf_idx = 0
    for buf in input_bufs:
        if buf_idx in packed_constants:
            # Skip — will be set as constant
            pass
        encoder.setBuffer_offset_atIndex_(buf, 0, buf_idx)
        buf_idx += 1

    # Bind output buffers
    for buf in output_bufs:
        if buf_idx in packed_constants:
            pass
        encoder.setBuffer_offset_atIndex_(buf, 0, buf_idx)
        buf_idx += 1

    # Bind constants (setBytes for small data)
    for idx, data in packed_constants.items():
        encoder.setBytes_length_atIndex_(data, len(data), idx)

    # Dispatch
    grid_size = Metal.MTLSizeMake(*grid)
    tg_size = Metal.MTLSizeMake(*threadgroup)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)

    # Finish
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    # Check for errors
    if cmd_buf.error() is not None:
        raise RuntimeError(f"Metal kernel execution failed: {cmd_buf.error()}")

    # Read output buffers back to JAX arrays
    results = []
    for buf, shape, dtype, nbytes in zip(output_bufs, output_shapes, output_dtypes, output_nbytes, strict=False):
        np_dtype = np.dtype(dtype)
        # PyObjC's contents() returns an objc.varlist; use as_buffer() to get memoryview
        membuf = buf.contents().as_buffer(nbytes)
        arr = np.frombuffer(membuf, dtype=np_dtype).copy().reshape(shape)
        results.append(jnp.array(arr))

    return results if len(results) > 1 else results[0]


# ──────────────────────────────────────────────────────────────────
# MetalKernel class (cached, reusable)
# ──────────────────────────────────────────────────────────────────


class MetalKernel:
    """A compiled Metal kernel that can be called repeatedly.

    Example:
        >>> kernel = MetalKernel('''
        ...     #include <metal_stdlib>
        ...     using namespace metal;
        ...     kernel void relu(device const float* x [[buffer(0)]],
        ...                      device float* y [[buffer(1)]],
        ...                      uint gid [[thread_position_in_grid]]) {
        ...         y[gid] = max(0.0f, x[gid]);
        ...     }
        ... ''', "relu")
        >>> y = kernel(inputs=[x], output_shapes=[(n,)], output_dtypes=[jnp.float32],
        ...            grid=(n, 1, 1), threadgroup=(256, 1, 1))
    """

    def __init__(self, source: str, function_name: str):
        self.source = source
        self.function_name = function_name
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = _get_pipeline(self.source, self.function_name)
        return self._pipeline

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self.pipeline.maxTotalThreadsPerThreadgroup()

    @property
    def thread_execution_width(self) -> int:
        return self.pipeline.threadExecutionWidth()

    def __call__(
        self,
        *,
        inputs: Sequence[jax.Array] = (),
        output_shapes: Sequence[tuple[int, ...]] = (),
        output_dtypes: Sequence = (),
        grid: tuple[int, int, int] = (1, 1, 1),
        threadgroup: tuple[int, int, int] = (1, 1, 1),
        constants: dict[int, object] | None = None,
    ) -> list[jax.Array] | jax.Array:
        return execute(
            self.source,
            self.function_name,
            inputs=inputs,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            grid=grid,
            threadgroup=threadgroup,
            constants=constants,
        )

    def __repr__(self):
        return f"MetalKernel('{self.function_name}', max_tg={self.max_threads_per_threadgroup})"


# ──────────────────────────────────────────────────────────────────
# Decorator API
# ──────────────────────────────────────────────────────────────────


def kernel(
    function_name: str,
    *,
    grid: tuple[int, int, int] | Callable = (1, 1, 1),
    threadgroup: tuple[int, int, int] = (256, 1, 1),
):
    """Decorator to define a Metal kernel from a source-returning function.

    Example:
        >>> @metal.kernel("my_relu", grid=lambda n: (n, 1, 1), threadgroup=(256, 1, 1))
        ... def relu():
        ...     return '''
        ...     #include <metal_stdlib>
        ...     using namespace metal;
        ...     kernel void my_relu(device const float* x [[buffer(0)]],
        ...                         device float* y [[buffer(1)]],
        ...                         uint gid [[thread_position_in_grid]]) {
        ...         y[gid] = max(0.0f, x[gid]);
        ...     }
        ...     '''
        >>> y = relu(x, output_shapes=[(1024,)], output_dtypes=[jnp.float32])
    """

    def decorator(fn):
        source = fn()
        compiled = MetalKernel(source, function_name)

        def wrapper(*inputs, output_shapes, output_dtypes, constants=None, **kwargs):
            g = grid
            if callable(g):
                # Compute grid from first input's size
                n = int(np.prod(inputs[0].shape)) if inputs else 1
                g = g(n)
                if isinstance(g, int):
                    g = (g, 1, 1)
            tg = kwargs.get("threadgroup", threadgroup)
            return compiled(
                inputs=inputs,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                grid=g,
                threadgroup=tg,
                constants=constants,
            )

        wrapper.kernel = compiled
        wrapper.__name__ = function_name
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


# ──────────────────────────────────────────────────────────────────
# Utility: device info
# ──────────────────────────────────────────────────────────────────


def device_info() -> dict:
    """Return Metal device information."""
    device, _ = _ensure_metal()
    return {
        "name": str(device.name()),
        "has_unified_memory": bool(device.hasUnifiedMemory()),
        "max_buffer_length": int(device.maxBufferLength()),
        "max_threadgroup_memory": int(device.maxThreadgroupMemoryLength()),
        "max_threads_per_threadgroup": int(device.maxThreadsPerThreadgroup().width),
        "recommended_max_working_set_size": int(device.recommendedMaxWorkingSetSize()),
    }
