from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .cases import BenchCase, register_case


def _attn_block(x: jnp.ndarray, wqkv: jnp.ndarray) -> jnp.ndarray:
    # x: [B, T, D], wqkv: [D, 3D]
    B, T, D = x.shape
    H = 8
    Dh = D // H

    qkv = x @ wqkv  # [B, T, 3D]
    qkv = qkv.reshape((B, T, 3, H, Dh))
    q = jnp.swapaxes(qkv[:, :, 0, :, :], 1, 2)  # [B, H, T, Dh]
    k = jnp.swapaxes(qkv[:, :, 1, :, :], 1, 2)
    v = jnp.swapaxes(qkv[:, :, 2, :, :], 1, 2)

    scale = jnp.asarray(Dh, dtype=x.dtype) ** jnp.asarray(-0.5, dtype=x.dtype)
    logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    return out


def _conv_resnetish(x: jnp.ndarray, k1: jnp.ndarray, k2: jnp.ndarray, k3: jnp.ndarray) -> jnp.ndarray:
    y = lax.conv_general_dilated(
        x,
        k1,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    y = jax.nn.relu(y)
    y = lax.conv_general_dilated(
        y,
        k2,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    y = jax.nn.relu(y)
    y = lax.conv_general_dilated(
        y,
        k3,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return x + y


def _embedding_loss(emb: jnp.ndarray, idx: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    gathered = jnp.take(emb, idx, axis=0)  # [batch, dim]
    return (gathered * w).sum()


def _layernorm_softmaxish(x: jnp.ndarray) -> jnp.ndarray:
    # x: [B, T, D]
    mean = x.mean(axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + jnp.asarray(1e-5, x.dtype))
    # Softmax over T (common attention-like reduction pattern)
    logits = y.mean(axis=-1)  # [B, T]
    return jax.nn.softmax(logits, axis=-1)


def _decode_slice_update(x: jnp.ndarray) -> jnp.ndarray:
    # x: [T, D]
    T, D = x.shape
    init = jnp.zeros((T, D), dtype=x.dtype)

    def step(buf, t):
        row = lax.dynamic_slice(buf, (t, 0), (1, D))  # [1, D]
        row = row.reshape((D,))
        new_row = jnp.tanh(row + x[t])
        buf2 = lax.dynamic_update_slice(buf, new_row.reshape((1, D)), (t, 0))
        return buf2, new_row.sum()

    ts = jnp.arange(T, dtype=jnp.int32)
    buf, sums = lax.scan(step, init, ts)
    return buf + sums.mean()


def _register() -> None:
    # Shapes fixed per perf-and-impl.md anchors.
    register_case(
        BenchCase(
            name="anchor.attn_block",
            description="QKV projection + scaled dot-product attention (B=2, T=128, D=256, H=8)",
            build=lambda rng: (
                _attn_block,
                (
                    jnp.asarray(rng.standard_normal((2, 128, 256), dtype=np.float32)),
                    jnp.asarray(rng.standard_normal((256, 3 * 256), dtype=np.float32)),
                ),
                {},
            ),
        )
    )
    register_case(
        BenchCase(
            name="anchor.conv_resnetish",
            description="3 conv blocks + residual (B=2, 64x64, C=32)",
            build=lambda rng: (
                _conv_resnetish,
                (
                    jnp.asarray(rng.standard_normal((2, 64, 64, 32), dtype=np.float32)),
                    jnp.asarray(rng.standard_normal((3, 3, 32, 32), dtype=np.float32)),
                    jnp.asarray(rng.standard_normal((3, 3, 32, 32), dtype=np.float32)),
                    jnp.asarray(rng.standard_normal((3, 3, 32, 32), dtype=np.float32)),
                ),
                {},
            ),
        )
    )
    register_case(
        BenchCase(
            name="anchor.embedding_gather_scatter",
            description="embedding take + grad scatter (vocab=65536, dim=256, batch=4096)",
            build=lambda rng: (
                jax.grad(_embedding_loss),
                (
                    jnp.asarray(rng.standard_normal((65536, 256), dtype=np.float32)),
                    jnp.asarray(rng.integers(0, 65536, size=(4096,), dtype=np.int32)),
                    jnp.asarray(rng.standard_normal((4096, 256), dtype=np.float32)),
                ),
                {},
            ),
        )
    )
    register_case(
        BenchCase(
            name="anchor.layernorm_softmaxish",
            description="layernorm-like reductions + softmax (B=16, T=256, D=512)",
            build=lambda rng: (
                _layernorm_softmaxish,
                (jnp.asarray(rng.standard_normal((16, 256, 512), dtype=np.float32)),),
                {},
            ),
        )
    )
    register_case(
        BenchCase(
            name="anchor.decode_slice_update",
            description="scan of dynamic_slice/update_slice (T=1024, D=256)",
            build=lambda rng: (
                _decode_slice_update,
                (jnp.asarray(rng.standard_normal((1024, 256), dtype=np.float32)),),
                {},
            ),
        )
    )


_register()
