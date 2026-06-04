from __future__ import annotations

import jax
import jax.numpy as jnp


def _stablehlo_text(lowered) -> str:
    try:
        return str(lowered.compiler_ir("stablehlo"))
    except Exception:
        return str(lowered.compiler_ir(dialect="stablehlo"))


def _probe_one(name: str, fn, *args) -> dict[str, object]:
    lowered = jax.jit(fn).lower(*args)
    text = _stablehlo_text(lowered)
    return {
        "name": name,
        "has_stablehlo_rng": "stablehlo.rng" in text,
        "has_stablehlo_rng_bit_generator": "stablehlo.rng_bit_generator" in text,
    }


def main() -> None:
    key = jax.random.key(0)
    probes = [
        ("random.uniform.f32", lambda k: jax.random.uniform(k, (8, 8), dtype=jnp.float32), key),
        ("random.normal.f32", lambda k: jax.random.normal(k, (8, 8), dtype=jnp.float32), key),
        ("random.bernoulli", lambda k: jax.random.bernoulli(k, p=0.3, shape=(8, 8)), key),
        ("random.randint.i32", lambda k: jax.random.randint(k, (8, 8), 0, 1024, dtype=jnp.int32), key),
    ]

    any_rng = False
    any_rngbg = False
    for name, fn, *args in probes:
        rec = _probe_one(name, fn, *args)
        any_rng |= bool(rec["has_stablehlo_rng"])
        any_rngbg |= bool(rec["has_stablehlo_rng_bit_generator"])
        print(
            f"{rec['name']}: stablehlo.rng={rec['has_stablehlo_rng']} "
            f"stablehlo.rng_bit_generator={rec['has_stablehlo_rng_bit_generator']}"
        )

    print(f"any stablehlo.rng: {any_rng}")
    print(f"any stablehlo.rng_bit_generator: {any_rngbg}")


if __name__ == "__main__":
    main()
