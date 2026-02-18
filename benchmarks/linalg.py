from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from .cases import BenchCase, register_case


def _batched_cholesky(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.cholesky(a)


def _batched_triangular_solve(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return lax.linalg.triangular_solve(
        a,
        b,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
    )


def _make_pd_batch(rng: np.random.Generator, batch: int, n: int) -> np.ndarray:
    x = rng.standard_normal((batch, n, n), dtype=np.float32)
    a = x @ np.transpose(x, (0, 2, 1))
    a += np.eye(n, dtype=np.float32)[None, :, :] * 1e-3
    return a


def _make_lower_batch(rng: np.random.Generator, batch: int, n: int, dtype: np.dtype) -> np.ndarray:
    x = rng.standard_normal((batch, n, n), dtype=np.float32)
    a = np.tril(x)
    a += np.eye(n, dtype=np.float32)[None, :, :] * (n * 1e-2)
    return a.astype(dtype, copy=False)


def _register() -> None:
    register_case(
        BenchCase(
            name="linalg.batched_cholesky",
            description="batched cholesky (batch=16, n=64, float32)",
            build=lambda rng: (
                _batched_cholesky,
                (jnp.asarray(_make_pd_batch(rng, batch=16, n=64)),),
                {},
            ),
        )
    )

    register_case(
        BenchCase(
            name="linalg.batched_triangular_solve_f16",
            description="batched triangular_solve (batch=16, n=64, nrhs=32, float16)",
            build=lambda rng: (
                _batched_triangular_solve,
                (
                    jnp.asarray(_make_lower_batch(rng, batch=16, n=64, dtype=np.float16)),
                    jnp.asarray(rng.standard_normal((16, 64, 32), dtype=np.float32).astype(np.float16)),
                ),
                {},
            ),
        )
    )


_register()
