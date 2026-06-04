"""
Competitive benchmark: jax-metallib (this project) vs jax-mps (tillahoffmann/jax-mps).

Both register as JAX platform 'mps', so they can't coexist in one process.
This script spawns separate subprocesses using each venv's Python interpreter.

Usage:
    # From repo root:
    uv run python benchmarks/vs_jax_mps.py

    # Custom jax-mps venv:
    uv run python benchmarks/vs_jax_mps.py --jax-mps-python .venv-jax-mps/bin/python
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# Shared preamble injected into every subprocess
_PREAMBLE = textwrap.dedent(
    """\
    import jax
    import jax.numpy as jnp
    import numpy as np
    import time
    import json
    import sys

    def block_ready(x):
        jax.tree_util.tree_map(lambda l: l.block_until_ready() if hasattr(l, 'block_until_ready') else l, x)

    def bench(fn, args, kwargs=None, warmup=10, iters=50):
        kwargs = kwargs or {}
        jit_fn = jax.jit(fn)
        # Warmup: compile + first runs
        out = jit_fn(*args, **kwargs)
        block_ready(out)
        for _ in range(warmup):
            out = jit_fn(*args, **kwargs)
            block_ready(out)
        # Timed
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = jit_fn(*args, **kwargs)
            block_ready(out)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        import statistics
        return {
            "median_ms": statistics.median(times) * 1e3,
            "min_ms": min(times) * 1e3,
            "p90_ms": sorted(times)[int(len(times)*0.9)] * 1e3,
            "times_ms": [t * 1e3 for t in times],
        }
"""
)

BENCHMARKS: list[tuple[str, str, str]] = []


def _add(name: str, desc: str, source: str) -> None:
    BENCHMARKS.append((name, desc, textwrap.dedent(source)))


# ===== 1. Elementwise ops (large tensors) =====
_add(
    "elemwise.add_f32",
    "Elementwise add, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a, b: a + b, (x, y))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.add_f16",
    "Elementwise add, float16, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    result = bench(lambda a, b: a + b, (x, y))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.mul_f32",
    "Elementwise multiply, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a, b: a * b, (x, y))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.exp_f32",
    "Elementwise exp, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(jnp.exp, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.tanh_f32",
    "Elementwise tanh, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(jnp.tanh, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.sin_f32",
    "Elementwise sin, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(jnp.sin, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.sqrt_f32",
    "Elementwise sqrt, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(np.abs(rng.standard_normal((4096, 4096)).astype(np.float32)))
    result = bench(jnp.sqrt, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "elemwise.abs_f32",
    "Elementwise abs, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(jnp.abs, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

# ===== 2. Fused elementwise chains (this is where jax-metallib should shine) =====
_add(
    "fusion.add_mul_f32",
    "Chain: (x+y)*z, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    z = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a, b, c: (a + b) * c, (x, y, z))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "fusion.add_mul_f16",
    "Chain: (x+y)*z, float16, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    z = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    result = bench(lambda a, b, c: (a + b) * c, (x, y, z))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

_add(
    "fusion.exp_mul_add_f32",
    "Chain: exp(x)*y+z, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32) * 0.1)
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    z = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a, b, c: jnp.exp(a) * b + c, (x, y, z))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "fusion.gelu_approx_f32",
    "GELU approx: x*sigmoid(1.702*x), float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    def gelu_approx(x):
        return x * jax.nn.sigmoid(1.702 * x)
    result = bench(gelu_approx, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "fusion.5op_chain_f32",
    "5-op chain: sub(exp(add(tanh(mul(x,y)),z)),w), float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32) * 0.5)
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32) * 0.5)
    z = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    w = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    def chain5(x, y, z, w):
        t1 = x * y
        t2 = jnp.tanh(t1)
        t3 = t2 + z
        t4 = jnp.exp(t3 * 0.1)
        return t4 - w
    result = bench(chain5, (x, y, z, w))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "fusion.5op_chain_f16",
    "5-op chain: sub(exp(add(tanh(mul(x,y)),z)),w), float16, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array((rng.standard_normal((4096, 4096)) * 0.5).astype(np.float32).astype(np.float16))
    y = jnp.array((rng.standard_normal((4096, 4096)) * 0.5).astype(np.float32).astype(np.float16))
    z = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    w = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    def chain5(x, y, z, w):
        t1 = x * y
        t2 = jnp.tanh(t1)
        t3 = t2 + z
        t4 = jnp.exp(t3 * jnp.float16(0.1))
        return t4 - w
    result = bench(chain5, (x, y, z, w))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

_add(
    "fusion.diamond_f32",
    "Diamond: (x+y)*(x-y), float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    y = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a, b: (a + b) * (a - b), (x, y))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "fusion.softmax_numerator_f32",
    "Softmax numerator: exp(x - max(x)), float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    def softmax_num(x):
        return jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    result = bench(softmax_num, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

# ===== 3. Matmul / Linear algebra =====
_add(
    "linalg.matmul_f32",
    "Matmul [1024,1024]@[1024,1024], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1024, 1024)).astype(np.float32))
    y = jnp.array(rng.standard_normal((1024, 1024)).astype(np.float32))
    result = bench(lambda a, b: a @ b, (x, y))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "linalg.matmul_f16",
    "Matmul [1024,1024]@[1024,1024], float16",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1024, 1024)).astype(np.float32).astype(np.float16))
    y = jnp.array(rng.standard_normal((1024, 1024)).astype(np.float32).astype(np.float16))
    result = bench(lambda a, b: a @ b, (x, y))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

_add(
    "linalg.matmul_large_f32",
    "Matmul [2048,2048]@[2048,2048], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((2048, 2048)).astype(np.float32))
    y = jnp.array(rng.standard_normal((2048, 2048)).astype(np.float32))
    result = bench(lambda a, b: a @ b, (x, y))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "linalg.batched_matmul_f32",
    "Batched matmul [4,16,256,64]@[4,16,64,256], float32",
    """
    rng = np.random.default_rng(42)
    q = jnp.array(rng.standard_normal((4, 16, 256, 64)).astype(np.float32))
    k = jnp.array(rng.standard_normal((4, 16, 256, 64)).astype(np.float32))
    result = bench(lambda a, b: jnp.matmul(a, jnp.swapaxes(b, -2, -1)), (q, k))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

# ===== 4. Reductions =====
_add(
    "reduce.sum_f32",
    "Reduce sum axis=-1, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a: jnp.sum(a, axis=-1), (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "reduce.max_f32",
    "Reduce max axis=-1, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a: jnp.max(a, axis=-1), (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "reduce.mean_f32",
    "Reduce mean axis=-1, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(lambda a: jnp.mean(a, axis=-1), (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

# ===== 5. Softmax (composite) =====
_add(
    "composite.softmax_f32",
    "Full softmax, float32, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32))
    result = bench(jax.nn.softmax, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "composite.softmax_f16",
    "Full softmax, float16, 4096x4096",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 4096)).astype(np.float32).astype(np.float16))
    result = bench(jax.nn.softmax, (x,))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

# ===== 6. Convolution =====
_add(
    "conv.2d_f32",
    "Conv2D 3x3, NHWC [8,128,128,64]->[8,128,128,64], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((8, 128, 128, 64)).astype(np.float32))
    w = jnp.array(rng.standard_normal((3, 3, 64, 64)).astype(np.float32))
    def conv2d(a, k):
        return jax.lax.conv_general_dilated(a, k, (1,1), 'SAME',
            dimension_numbers=('NHWC','HWIO','NHWC'))
    result = bench(conv2d, (x, w))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "conv.2d_f16",
    "Conv2D 3x3, NHWC [8,128,128,64]->[8,128,128,64], float16",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((8, 128, 128, 64)).astype(np.float32).astype(np.float16))
    w = jnp.array(rng.standard_normal((3, 3, 64, 64)).astype(np.float32).astype(np.float16))
    def conv2d(a, k):
        return jax.lax.conv_general_dilated(a, k, (1,1), 'SAME',
            dimension_numbers=('NHWC','HWIO','NHWC'))
    result = bench(conv2d, (x, w))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

# ===== 7. Gather/Scatter =====
_add(
    "memory.gather_f32",
    "Embedding gather [65536,512] idx=8192, float32",
    """
    rng = np.random.default_rng(42)
    emb = jnp.array(rng.standard_normal((65536, 512)).astype(np.float32))
    idx = jnp.array(rng.integers(0, 65536, size=(8192,), dtype=np.int32))
    result = bench(lambda e, i: jnp.take(e, i, axis=0), (emb, idx))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

# ===== 8. Sort =====
_add(
    "sort.sort_f32",
    "Sort axis=-1, [4096,256], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((4096, 256)).astype(np.float32))
    result = bench(lambda a: jax.lax.sort(a, dimension=-1), (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

# ===== 9. Real-world patterns =====
_add(
    "pattern.layernorm_f32",
    "LayerNorm [16,256,512], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((16, 256, 512)).astype(np.float32))
    def layernorm(x):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + 1e-5)
    result = bench(layernorm, (x,))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "pattern.attention_f32",
    "Scaled dot-product attention [2,8,128,32], float32",
    """
    rng = np.random.default_rng(42)
    B, H, T, Dh = 2, 8, 128, 32
    q = jnp.array(rng.standard_normal((B, H, T, Dh)).astype(np.float32))
    k = jnp.array(rng.standard_normal((B, H, T, Dh)).astype(np.float32))
    v = jnp.array(rng.standard_normal((B, H, T, Dh)).astype(np.float32))
    def attention(q, k, v):
        scale = jnp.float32(Dh) ** jnp.float32(-0.5)
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.matmul(weights, v)
    result = bench(attention, (q, k, v))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "pattern.attention_f16",
    "Scaled dot-product attention [2,8,128,32], float16",
    """
    rng = np.random.default_rng(42)
    B, H, T, Dh = 2, 8, 128, 32
    q = jnp.array(rng.standard_normal((B, H, T, Dh)).astype(np.float16))
    k = jnp.array(rng.standard_normal((B, H, T, Dh)).astype(np.float16))
    v = jnp.array(rng.standard_normal((B, H, T, Dh)).astype(np.float16))
    def attention(q, k, v):
        scale = jnp.float16(Dh) ** jnp.float16(-0.5)
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.matmul(weights, v)
    result = bench(attention, (q, k, v))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

_add(
    "pattern.conv_relu_block_f32",
    "3xConv+ReLU+residual [2,64,64,32], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((2, 64, 64, 32)).astype(np.float32))
    k1 = jnp.array(rng.standard_normal((3, 3, 32, 32)).astype(np.float32))
    k2 = jnp.array(rng.standard_normal((3, 3, 32, 32)).astype(np.float32))
    k3 = jnp.array(rng.standard_normal((3, 3, 32, 32)).astype(np.float32))
    def conv_block(x, k1, k2, k3):
        y = jax.lax.conv_general_dilated(x, k1, (1,1), 'SAME',
            dimension_numbers=('NHWC','HWIO','NHWC'))
        y = jax.nn.relu(y)
        y = jax.lax.conv_general_dilated(y, k2, (1,1), 'SAME',
            dimension_numbers=('NHWC','HWIO','NHWC'))
        y = jax.nn.relu(y)
        y = jax.lax.conv_general_dilated(y, k3, (1,1), 'SAME',
            dimension_numbers=('NHWC','HWIO','NHWC'))
        return x + y
    result = bench(conv_block, (x, k1, k2, k3))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "pattern.mlp_block_f32",
    "MLP: Linear(512->2048)->GELU->Linear(2048->512), float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((16, 256, 512)).astype(np.float32))
    w1 = jnp.array(rng.standard_normal((512, 2048)).astype(np.float32) * 0.02)
    w2 = jnp.array(rng.standard_normal((2048, 512)).astype(np.float32) * 0.02)
    def mlp(x, w1, w2):
        h = x @ w1
        h = jax.nn.gelu(h)
        return h @ w2
    result = bench(mlp, (x, w1, w2))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "pattern.mlp_block_f16",
    "MLP: Linear(512->2048)->GELU->Linear(2048->512), float16",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((16, 256, 512)).astype(np.float32).astype(np.float16))
    w1 = jnp.array((rng.standard_normal((512, 2048)) * 0.02).astype(np.float32).astype(np.float16))
    w2 = jnp.array((rng.standard_normal((2048, 512)) * 0.02).astype(np.float32).astype(np.float16))
    def mlp(x, w1, w2):
        h = x @ w1
        h = jax.nn.gelu(h)
        return h @ w2
    result = bench(mlp, (x, w1, w2))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

_add(
    "pattern.transformer_layer_f32",
    "Transformer layer (attn+MLP+LN) [2,128,256], float32",
    """
    rng = np.random.default_rng(42)
    B, T, D, H = 2, 128, 256, 8
    Dh = D // H
    x = jnp.array(rng.standard_normal((B, T, D)).astype(np.float32))
    wqkv = jnp.array((rng.standard_normal((D, 3*D)) * 0.02).astype(np.float32))
    wo = jnp.array((rng.standard_normal((D, D)) * 0.02).astype(np.float32))
    w1 = jnp.array((rng.standard_normal((D, 4*D)) * 0.02).astype(np.float32))
    w2 = jnp.array((rng.standard_normal((4*D, D)) * 0.02).astype(np.float32))

    def transformer_layer(x, wqkv, wo, w1, w2):
        # Pre-norm
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xn = (x - mean) * jax.lax.rsqrt(var + 1e-5)

        # Multi-head attention
        qkv = xn @ wqkv
        qkv = qkv.reshape((B, T, 3, H, Dh))
        q = jnp.swapaxes(qkv[:, :, 0, :, :], 1, 2)
        k = jnp.swapaxes(qkv[:, :, 1, :, :], 1, 2)
        v = jnp.swapaxes(qkv[:, :, 2, :, :], 1, 2)
        scale = jnp.float32(Dh) ** jnp.float32(-0.5)
        logits = jnp.einsum('bhtd,bhsd->bhts', q, k) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        attn_out = jnp.einsum('bhts,bhsd->bhtd', weights, v)
        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape((B, T, D))
        x = x + attn_out @ wo

        # Pre-norm + MLP
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xn = (x - mean) * jax.lax.rsqrt(var + 1e-5)
        h = xn @ w1
        h = jax.nn.gelu(h)
        x = x + h @ w2
        return x

    result = bench(transformer_layer, (x, wqkv, wo, w1, w2))
    result["dtype"] = "float32"
    print(json.dumps(result))
""",
)

_add(
    "pattern.transformer_layer_f16",
    "Transformer layer (attn+MLP+LN) [2,128,256], float16",
    """
    rng = np.random.default_rng(42)
    B, T, D, H = 2, 128, 256, 8
    Dh = D // H
    x = jnp.array(rng.standard_normal((B, T, D)).astype(np.float32).astype(np.float16))
    wqkv = jnp.array((rng.standard_normal((D, 3*D)) * 0.02).astype(np.float32).astype(np.float16))
    wo = jnp.array((rng.standard_normal((D, D)) * 0.02).astype(np.float32).astype(np.float16))
    w1 = jnp.array((rng.standard_normal((D, 4*D)) * 0.02).astype(np.float32).astype(np.float16))
    w2 = jnp.array((rng.standard_normal((4*D, D)) * 0.02).astype(np.float32).astype(np.float16))

    def transformer_layer(x, wqkv, wo, w1, w2):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xn = (x - mean) * jax.lax.rsqrt(var + jnp.float16(1e-5))

        qkv = xn @ wqkv
        qkv = qkv.reshape((B, T, 3, H, Dh))
        q = jnp.swapaxes(qkv[:, :, 0, :, :], 1, 2)
        k = jnp.swapaxes(qkv[:, :, 1, :, :], 1, 2)
        v = jnp.swapaxes(qkv[:, :, 2, :, :], 1, 2)
        scale = jnp.float16(Dh) ** jnp.float16(-0.5)
        logits = jnp.einsum('bhtd,bhsd->bhts', q, k) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        attn_out = jnp.einsum('bhts,bhsd->bhtd', weights, v)
        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape((B, T, D))
        x = x + attn_out @ wo

        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xn = (x - mean) * jax.lax.rsqrt(var + jnp.float16(1e-5))
        h = xn @ w1
        h = jax.nn.gelu(h)
        x = x + h @ w2
        return x

    result = bench(transformer_layer, (x, wqkv, wo, w1, w2))
    result["dtype"] = "float16"
    print(json.dumps(result))
""",
)

# ===== 10. Full transformer models (multi-layer, realistic GPT-2 configs) =====


# Helper to generate transformer model benchmark source
def _transformer_model_source(B, T, D, H, n_layers, dtype):
    cast = ".astype(np.float16)" if dtype == "float16" else ""
    eps_val = "jnp.float16(1e-5)" if dtype == "float16" else "1e-5"
    scale_dt = "jnp.float16" if dtype == "float16" else "jnp.float32"
    Dh = D // H
    ff = 4 * D
    return f"""
    rng = np.random.default_rng(42)
    B, T, D, H, Dh, n_layers, ff = {B}, {T}, {D}, {H}, {Dh}, {n_layers}, {ff}
    x = jnp.array(rng.standard_normal((B, T, D)).astype(np.float32){cast})
    flat_wqkv = jnp.array((rng.standard_normal(({n_layers}, D, 3*D)) * 0.02).astype(np.float32){cast})
    flat_wo   = jnp.array((rng.standard_normal(({n_layers}, D, D))   * 0.02).astype(np.float32){cast})
    flat_w1   = jnp.array((rng.standard_normal(({n_layers}, D, ff))  * 0.02).astype(np.float32){cast})
    flat_w2   = jnp.array((rng.standard_normal(({n_layers}, ff, D))  * 0.02).astype(np.float32){cast})

    def layernorm(x):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + {eps_val})

    def transformer_block(x, wqkv, wo, w1, w2):
        xn = layernorm(x)
        qkv = xn @ wqkv
        qkv = qkv.reshape((B, T, 3, H, Dh))
        q = jnp.swapaxes(qkv[:, :, 0, :, :], 1, 2)
        k = jnp.swapaxes(qkv[:, :, 1, :, :], 1, 2)
        v = jnp.swapaxes(qkv[:, :, 2, :, :], 1, 2)
        scale = {scale_dt}(Dh) ** {scale_dt}(-0.5)
        logits = jnp.einsum('bhtd,bhsd->bhts', q, k) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        attn_out = jnp.einsum('bhts,bhsd->bhtd', weights, v)
        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape((B, T, D))
        x = x + attn_out @ wo
        xn = layernorm(x)
        h = jax.nn.gelu(xn @ w1)
        x = x + h @ w2
        return x

    def forward(x, wqkv_all, wo_all, w1_all, w2_all):
        def body(carry, layer_idx):
            return transformer_block(carry, wqkv_all[layer_idx], wo_all[layer_idx],
                                     w1_all[layer_idx], w2_all[layer_idx]), None
        x, _ = jax.lax.scan(body, x, jnp.arange(n_layers))
        return layernorm(x)

    result = bench(forward, (x, flat_wqkv, flat_wo, flat_w1, flat_w2), warmup=5, iters=30)
    result["dtype"] = "{dtype}"
    print(json.dumps(result))
"""


_add(
    "model.gpt2_small_f32",
    "GPT-2 Small 4-layer [1,128,256], float32",
    _transformer_model_source(1, 128, 256, 8, 4, "float32"),
)

_add(
    "model.gpt2_small_f16",
    "GPT-2 Small 4-layer [1,128,256], float16",
    _transformer_model_source(1, 128, 256, 8, 4, "float16"),
)

_add(
    "model.gpt2_medium_f32",
    "GPT-2 Medium 6-layer [1,256,512], float32",
    _transformer_model_source(1, 256, 512, 8, 6, "float32"),
)

_add(
    "model.gpt2_medium_f16",
    "GPT-2 Medium 6-layer [1,256,512], float16",
    _transformer_model_source(1, 256, 512, 8, 6, "float16"),
)

_add(
    "model.gpt2_large_f16",
    "GPT-2 Large 6-layer [1,512,768], float16",
    _transformer_model_source(1, 512, 768, 12, 6, "float16"),
)

_add(
    "model.gpt2_batch_f16",
    "GPT-2 Small batched 4-layer [4,128,256], float16",
    _transformer_model_source(4, 128, 256, 8, 4, "float16"),
)


def run_one_benchmark(
    python_bin: str,
    bench_name: str,
    bench_source: str,
    label: str,
    timeout: int = 120,
) -> dict | None:
    full_source = _PREAMBLE + bench_source
    try:
        proc = subprocess.run(
            [python_bin, "-c", full_source],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/usr/local/bin", "HOME": str(Path.home())},
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            # Filter out just the error, skip warnings
            err_lines = [l for l in stderr.split("\n") if "WARNING" not in l and "experimental" not in l]
            err_msg = "\n".join(err_lines[-3:]) if err_lines else stderr[-200:]
            print(f"  [{label}] FAIL: {err_msg}", file=sys.stderr)
            return None
        # Parse last JSON line from stdout
        for line in reversed(proc.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        print(f"  [{label}] FAIL: no JSON output", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT after {timeout}s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [{label}] ERROR: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="jax-metallib vs jax-mps benchmark")
    parser.add_argument(
        "--jax-metallib-python",
        default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python interpreter with jax-metallib installed",
    )
    parser.add_argument(
        "--jax-mps-python",
        default=str(REPO_ROOT / ".venv-jax-mps" / "bin" / "python"),
        help="Python interpreter with jax-mps installed",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run benchmarks matching this substring",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per benchmark in seconds",
    )
    args = parser.parse_args()

    # Verify interpreters exist
    for label, path in [("jax-metallib", args.jax_metallib_python), ("jax-mps", args.jax_mps_python)]:
        if not Path(path).exists():
            print(f"ERROR: {label} Python not found at {path}")
            print(f"  Install {label} in the appropriate venv first.")
            sys.exit(1)

    # Get version info
    for label, pybin in [("jax-metallib", args.jax_metallib_python), ("jax-mps", args.jax_mps_python)]:
        try:
            out = subprocess.check_output(
                [pybin, "-c", "import jax; print(jax.__version__)"],
                text=True,
                timeout=10,
                env={"PATH": "/usr/bin:/usr/local/bin", "HOME": str(Path.home())},
            ).strip()
            print(f"{label}: JAX {out}")
        except Exception as e:
            print(f"{label}: version check failed: {e}")

    benchmarks = BENCHMARKS
    if args.filter:
        benchmarks = [(n, d, s) for n, d, s in benchmarks if args.filter in n]

    print(f"\nRunning {len(benchmarks)} benchmarks...\n")
    print(f"{'Benchmark':<40} {'jax-metallib (ms)':>18} {'jax-mps (ms)':>18} {'Speedup':>10}")
    print("-" * 90)

    results = []
    for name, desc, source in benchmarks:
        sys.stdout.write(f"{name:<40} ")
        sys.stdout.flush()

        silicon_result = run_one_benchmark(args.jax_metallib_python, name, source, "silicon", args.timeout)
        mps_result = run_one_benchmark(args.jax_mps_python, name, source, "jax-mps", args.timeout)

        silicon_ms = silicon_result["median_ms"] if silicon_result else None
        mps_ms = mps_result["median_ms"] if mps_result else None

        silicon_str = f"{silicon_ms:>14.3f} ms" if silicon_ms is not None else "         FAIL  "
        mps_str = f"{mps_ms:>14.3f} ms" if mps_ms is not None else "         FAIL  "

        if silicon_ms is not None and mps_ms is not None:
            speedup = mps_ms / silicon_ms
            if speedup >= 1.0:
                speedup_str = f"{speedup:>7.2f}x ✅"
            else:
                speedup_str = f"{speedup:>7.2f}x ❌"
        elif silicon_ms is not None and mps_ms is None:
            speedup_str = "  silicon ✅"
        elif silicon_ms is None and mps_ms is not None:
            speedup_str = "  jax-mps ✅"
        else:
            speedup_str = "  both FAIL"

        print(f"{silicon_str} {mps_str} {speedup_str}")

        results.append(
            {
                "name": name,
                "description": desc,
                "silicon": silicon_result,
                "jax_mps": mps_result,
                "speedup": (mps_ms / silicon_ms) if (silicon_ms and mps_ms) else None,
            }
        )

    # Summary
    print("\n" + "=" * 90)
    wins = sum(1 for r in results if r["speedup"] is not None and r["speedup"] >= 1.0)
    losses = sum(1 for r in results if r["speedup"] is not None and r["speedup"] < 1.0)
    silicon_only = sum(1 for r in results if r["silicon"] and not r["jax_mps"])
    mps_only = sum(1 for r in results if not r["silicon"] and r["jax_mps"])

    print(
        f"\nScoreboard:  jax-metallib wins: {wins}  |  jax-mps wins: {losses}  |  "
        f"silicon-only: {silicon_only}  |  mps-only: {mps_only}"
    )

    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    if speedups:
        geo_mean = np.exp(np.mean(np.log(speedups))) if speedups else 0
        print(f"Geometric mean speedup (silicon over jax-mps): {geo_mean:.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x  |  Min: {min(speedups):.2f}x")

    # Write detailed results
    out_dir = REPO_ROOT / "tmp-files"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    out_jsonl = out_dir / "vs_jax_mps.jsonl"
    with out_jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Markdown report
    out_md = out_dir / "vs_jax_mps.md"
    lines = []
    lines.append("# jax-metallib vs jax-mps Benchmark\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("## Results\n\n")
    lines.append("| Benchmark | Description | jax-metallib (ms) | jax-mps (ms) | Speedup | Winner |\n")
    lines.append("|-----------|-------------|------------------:|-------------:|--------:|--------|\n")

    for r in results:
        s_ms = f"{r['silicon']['median_ms']:.3f}" if r["silicon"] else "FAIL"
        m_ms = f"{r['jax_mps']['median_ms']:.3f}" if r["jax_mps"] else "FAIL"
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        if r["speedup"] is not None:
            winner = "jax-metallib" if r["speedup"] >= 1.0 else "jax-mps"
        elif r["silicon"] and not r["jax_mps"]:
            winner = "jax-metallib"
        elif not r["silicon"] and r["jax_mps"]:
            winner = "jax-mps"
        else:
            winner = "N/A"
        lines.append(f"| {r['name']} | {r['description']} | {s_ms} | {m_ms} | {sp} | {winner} |\n")

    if speedups:
        lines.append(f"\n**Geometric mean speedup: {geo_mean:.2f}x** (>1 = jax-metallib faster)\n\n")
        lines.append(f"**Wins:** jax-metallib {wins} / jax-mps {losses}\n")

    out_md.write_text("".join(lines))
    print(f"\nResults saved to: {out_jsonl} and {out_md}")


if __name__ == "__main__":
    import numpy as np  # only needed for geo mean in main

    main()
