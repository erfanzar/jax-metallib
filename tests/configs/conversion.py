from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig


def make_conversion_op_configs():
    with OperationTestConfig.module_name("conversion"):
        yield from [
            OperationTestConfig(
                lambda: jnp.arange(10, dtype=jnp.float32),
                differentiable_argnums=(),
            ),
            # float32 -> float16 has a gradient (cast back); compare it on both backends.
            OperationTestConfig(
                lambda x: x.astype(jnp.float16),
                lambda rng: rng.normal(size=(4, 5)),
            ),
            OperationTestConfig(
                lambda x: lax.bitcast_convert_type(x, jnp.int32),
                lambda rng: rng.normal(size=(4, 5)),
                differentiable_argnums=(),
            ),
        ]
