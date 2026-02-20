"""
Benchmark: jax-metallib (Apple GPU) vs JAX CPU backend.

Both backends can coexist via JAX_PLATFORMS, so we spawn subprocesses
with JAX_PLATFORMS=mps and JAX_PLATFORMS=cpu respectively.

Usage:
    # From repo root:
    .venv/bin/python benchmarks/vs_cpu.py

    # Only run attention benchmarks:
    .venv/bin/python benchmarks/vs_cpu.py --filter attention
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
        out = jit_fn(*args, **kwargs)
        block_ready(out)
        for _ in range(warmup):
            out = jit_fn(*args, **kwargs)
            block_ready(out)
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


# ===== 1. Elementwise ops =====
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

# ===== 2. Fused elementwise chains =====
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
    "Batched matmul [4,16,512,64]@[4,16,64,512], float32",
    """
    rng = np.random.default_rng(42)
    q = jnp.array(rng.standard_normal((4, 16, 512, 64)).astype(np.float32))
    k = jnp.array(rng.standard_normal((4, 16, 512, 64)).astype(np.float32))
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
    "Embedding gather [131072,768] idx=32768, float32",
    """
    rng = np.random.default_rng(42)
    emb = jnp.array(rng.standard_normal((131072, 768)).astype(np.float32))
    idx = jnp.array(rng.integers(0, 131072, size=(32768,), dtype=np.int32))
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
    "LayerNorm [16,512,1024], float32",
    """
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((16, 512, 1024)).astype(np.float32))
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
    "Scaled dot-product attention [4,16,512,64], float32",
    """
    rng = np.random.default_rng(42)
    B, H, T, Dh = 4, 16, 512, 64
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
    "Scaled dot-product attention [4,16,512,64], float16",
    """
    rng = np.random.default_rng(42)
    B, H, T, Dh = 4, 16, 512, 64
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
    "Transformer layer (attn+MLP+LN) [2,256,512], float32",
    """
    rng = np.random.default_rng(42)
    B, T, D, H = 2, 256, 512, 8
    Dh = D // H
    x = jnp.array(rng.standard_normal((B, T, D)).astype(np.float32))
    wqkv = jnp.array((rng.standard_normal((D, 3*D)) * 0.02).astype(np.float32))
    wo = jnp.array((rng.standard_normal((D, D)) * 0.02).astype(np.float32))
    w1 = jnp.array((rng.standard_normal((D, 4*D)) * 0.02).astype(np.float32))
    w2 = jnp.array((rng.standard_normal((4*D, D)) * 0.02).astype(np.float32))

    def transformer_layer(x, wqkv, wo, w1, w2):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xn = (x - mean) * jax.lax.rsqrt(var + 1e-5)

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
    "Transformer layer (attn+MLP+LN) [2,256,512], float16",
    """
    rng = np.random.default_rng(42)
    B, T, D, H = 2, 256, 512, 8
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_one_benchmark(
    python_bin: str,
    bench_name: str,
    bench_source: str,
    label: str,
    platform: str,
    timeout: int = 120,
) -> dict | None:
    full_source = _PREAMBLE + bench_source
    env = {
        "PATH": "/usr/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "JAX_PLATFORMS": platform,
    }
    try:
        proc = subprocess.run(
            [python_bin, "-c", full_source],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            err_lines = [l for l in stderr.split("\n") if "WARNING" not in l and "experimental" not in l]
            err_msg = "\n".join(err_lines[-3:]) if err_lines else stderr[-200:]
            print(f"  [{label}] FAIL: {err_msg}", file=sys.stderr)
            return None
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
    parser = argparse.ArgumentParser(description="jax-metallib (GPU) vs JAX CPU benchmark")
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python interpreter with jax-metallib installed",
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

    pybin = args.python
    if not Path(pybin).exists():
        print(f"ERROR: Python not found at {pybin}")
        sys.exit(1)

    try:
        out = subprocess.check_output(
            [pybin, "-c", "import jax; print(jax.__version__)"],
            text=True,
            timeout=10,
            env={"PATH": "/usr/bin:/usr/local/bin", "HOME": str(Path.home())},
        ).strip()
        print(f"JAX {out}")
    except Exception as e:
        print(f"version check failed: {e}")

    benchmarks = BENCHMARKS
    if args.filter:
        benchmarks = [(n, d, s) for n, d, s in benchmarks if args.filter in n]

    print(f"\nRunning {len(benchmarks)} benchmarks...\n")
    print(f"{'Benchmark':<40} {'GPU (ms)':>18} {'CPU (ms)':>18} {'Speedup':>10}")
    print("-" * 90)

    results = []
    for name, desc, source in benchmarks:
        sys.stdout.write(f"{name:<40} ")
        sys.stdout.flush()

        gpu_result = run_one_benchmark(pybin, name, source, "gpu", "mps", args.timeout)
        cpu_result = run_one_benchmark(pybin, name, source, "cpu", "cpu", args.timeout)

        gpu_ms = gpu_result["median_ms"] if gpu_result else None
        cpu_ms = cpu_result["median_ms"] if cpu_result else None

        gpu_str = f"{gpu_ms:>14.3f} ms" if gpu_ms is not None else "         FAIL  "
        cpu_str = f"{cpu_ms:>14.3f} ms" if cpu_ms is not None else "         FAIL  "

        if gpu_ms is not None and cpu_ms is not None:
            speedup = cpu_ms / gpu_ms
            if speedup >= 1.0:
                speedup_str = f"{speedup:>7.2f}x ✅"
            else:
                speedup_str = f"{speedup:>7.2f}x ❌"
        elif gpu_ms is not None and cpu_ms is None:
            speedup_str = "   gpu ✅"
        elif gpu_ms is None and cpu_ms is not None:
            speedup_str = "   cpu ✅"
        else:
            speedup_str = "  both FAIL"

        print(f"{gpu_str} {cpu_str} {speedup_str}")

        results.append(
            {
                "name": name,
                "description": desc,
                "gpu": gpu_result,
                "cpu": cpu_result,
                "speedup": (cpu_ms / gpu_ms) if (gpu_ms and cpu_ms) else None,
            }
        )

    print("\n" + "=" * 90)
    wins = sum(1 for r in results if r["speedup"] is not None and r["speedup"] >= 1.0)
    losses = sum(1 for r in results if r["speedup"] is not None and r["speedup"] < 1.0)
    gpu_only = sum(1 for r in results if r["gpu"] and not r["cpu"])
    cpu_only = sum(1 for r in results if not r["gpu"] and r["cpu"])

    print(
        f"\nScoreboard:  GPU wins: {wins}  |  CPU wins: {losses}  |  "
        f"gpu-only: {gpu_only}  |  cpu-only: {cpu_only}"
    )

    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    if speedups:
        geo_mean = np.exp(np.mean(np.log(speedups))) if speedups else 0
        print(f"Geometric mean speedup (GPU over CPU): {geo_mean:.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x  |  Min: {min(speedups):.2f}x")

    out_dir = REPO_ROOT / "tmp-files"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "vs_cpu.jsonl"
    with out_jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    out_md = out_dir / "vs_cpu.md"
    lines = []
    lines.append("# jax-metallib (GPU) vs JAX CPU Benchmark\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("## Results\n\n")
    lines.append("| Benchmark | Description | GPU (ms) | CPU (ms) | Speedup | Winner |\n")
    lines.append("|-----------|-------------|------------------:|-------------:|--------:|--------|\n")

    for r in results:
        g_ms = f"{r['gpu']['median_ms']:.3f}" if r["gpu"] else "FAIL"
        c_ms = f"{r['cpu']['median_ms']:.3f}" if r["cpu"] else "FAIL"
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        if r["speedup"] is not None:
            winner = "GPU" if r["speedup"] >= 1.0 else "CPU"
        elif r["gpu"] and not r["cpu"]:
            winner = "GPU"
        elif not r["gpu"] and r["cpu"]:
            winner = "CPU"
        else:
            winner = "N/A"
        lines.append(f"| {r['name']} | {r['description']} | {g_ms} | {c_ms} | {sp} | {winner} |\n")

    if speedups:
        lines.append(f"\n**Geometric mean speedup: {geo_mean:.2f}x** (>1 = GPU faster)\n\n")
        lines.append(f"**Wins:** GPU {wins} / CPU {losses}\n")

    out_md.write_text("".join(lines))
    print(f"\nResults saved to: {out_jsonl} and {out_md}")


if __name__ == "__main__":
    import numpy as np

    main()
