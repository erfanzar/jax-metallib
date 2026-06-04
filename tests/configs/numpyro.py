import jax
import numpy
from jax import numpy as jnp
from jax import random
from numpyro import distributions as dists

from .util import CPU_DEVICE, OperationTestConfig


class NumpyroDistributionTestConfig(OperationTestConfig):
    """
    Test config for numpyro distributions that evaluates log_prob and samples.

    The sample is generated on CPU from the exact same distribution parameters
    that will be used for log_prob, ensuring the sample is always in-support
    regardless of platform or execution order.

    Args:
        dist_cls: Distribution class to test.
        *args: Factory functions for distribution constructor parameters.
        **kwargs: Keyword arguments passed to OperationTestConfig.
    """

    def __init__(self, dist_cls: type[dists.Distribution], *args, **kwargs) -> None:
        kwargs.setdefault("name", dist_cls.__name__)
        self._dist_cls = dist_cls
        self._param_factories = args

        super().__init__(
            lambda x, *params: dist_cls(*params).log_prob(x).mean(),
            # Placeholder for sample (arg 0) — replaced in evaluate_value/grad.
            numpy.float32(0.0),
            *args,
            **kwargs,
        )

    def _generate_sample_and_args(self):
        """Generate args with a proper sample as arg[0].

        Uses the same seed to produce identical parameters, then builds a
        sample on CPU from plain numpy parameters to avoid any device-specific
        behaviour, before returning the full arg list.
        """
        rng = numpy.random.default_rng(self.seed)
        args = self.get_args(rng)
        kwargs = self.get_kwargs(rng)
        # Build the sample from plain numpy params on CPU to avoid device issues.
        numpy_params = [numpy.asarray(a) if hasattr(a, "__array__") else a for a in args[1:]]
        with jax.default_device(CPU_DEVICE):
            dist = self._dist_cls(*[jnp.asarray(p) if isinstance(p, numpy.ndarray) else p for p in numpy_params])
            sample = dist.sample(random.key(17))
        args[0] = jnp.asarray(numpy.asarray(sample))
        return args, kwargs

    def evaluate_value(self, jit: bool):
        args, kwargs = self._generate_sample_and_args()
        result = self.func(*args, **kwargs)
        return result

    def evaluate_grad(self, argnum: int, jit: bool):
        args, kwargs = self._generate_sample_and_args()

        func = self.func
        result = func(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            num_return_values = len(result)
        else:
            num_return_values = None

        grad_vals = []
        for returnnum in range(num_return_values or 1):

            def target(*args, returnnum=returnnum, **kwargs):
                result = func(*args, **kwargs)
                if num_return_values is None:
                    assert isinstance(result, jnp.ndarray), f"Output of '{func}' is not a tensor."
                else:
                    result = result[returnnum]
                if result.dtype == jnp.complex64:
                    result = jnp.abs(result)
                if result.shape != ():
                    result = result.mean()
                return result

            grad_func = self.grad_transform(target, argnums=argnum)
            grad_vals.append(grad_func(*args))
        return tuple(grad_vals)


def make_numpyro_op_configs():
    with OperationTestConfig.module_name("numpyro"):
        for batch_shape in [(), (3,)]:
            yield from [
                NumpyroDistributionTestConfig(
                    dists.Normal,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Gamma,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Exponential,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Uniform,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.standard_normal(bs) + 2,
                ),
                NumpyroDistributionTestConfig(
                    dists.Laplace,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Cauchy,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfNormal,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfCauchy,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.LogNormal,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Beta,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.StudentT,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs) + 2,  # df > 2
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Dirichlet,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, (*bs, 3)),
                ),
                # Discrete distributions (differentiable_argnums excludes sample at 0).
                NumpyroDistributionTestConfig(
                    dists.BernoulliProbs,
                    lambda rng, bs=batch_shape: rng.uniform(0.1, 0.9, bs),
                    differentiable_argnums=(1,),
                    name="Bernoulli",
                ),
                NumpyroDistributionTestConfig(
                    dists.BinomialProbs,
                    lambda rng, bs=batch_shape: rng.uniform(0.1, 0.9, bs),
                    10,  # total_count (not differentiable)
                    differentiable_argnums=(1,),
                    name="Binomial",
                ),
                NumpyroDistributionTestConfig(
                    dists.CategoricalProbs,
                    lambda rng, bs=batch_shape: rng.dirichlet(numpy.ones(5), bs),
                    differentiable_argnums=(1,),
                    name="Categorical",
                ),
                NumpyroDistributionTestConfig(
                    dists.Poisson,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                    differentiable_argnums=(1,),
                ),
                NumpyroDistributionTestConfig(
                    dists.GeometricProbs,
                    lambda rng, bs=batch_shape: rng.uniform(0.1, 0.9, bs),
                    differentiable_argnums=(1,),
                    name="Geometric",
                ),
                NumpyroDistributionTestConfig(
                    dists.NegativeBinomial2,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # mean
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                    differentiable_argnums=(1, 2),
                ),
                NumpyroDistributionTestConfig(
                    dists.MultinomialProbs,
                    lambda rng, bs=batch_shape: rng.dirichlet(numpy.ones(5), bs),
                    10,  # total_count (not differentiable)
                    differentiable_argnums=(1,),
                    name="Multinomial",
                ),
                # Additional continuous distributions.
                NumpyroDistributionTestConfig(
                    dists.Gumbel,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Logistic,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Pareto,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # scale
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # alpha
                ),
                NumpyroDistributionTestConfig(
                    dists.Weibull,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # scale
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                ),
                NumpyroDistributionTestConfig(
                    dists.Chi2,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs) + 2,  # df
                ),
                NumpyroDistributionTestConfig(
                    dists.InverseGamma,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # rate
                ),
                NumpyroDistributionTestConfig(
                    dists.VonMises,
                    lambda rng, bs=batch_shape: rng.uniform(-numpy.pi, numpy.pi, bs),  # loc
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                ),
                # Multivariate distributions.
                OperationTestConfig(
                    lambda x, loc, covariance_matrix, precision_matrix, scale_tril: dists.MultivariateNormal(
                        loc, covariance_matrix, precision_matrix, scale_tril
                    )
                    .log_prob(x)
                    .mean(),
                    lambda rng, bs=batch_shape: rng.standard_normal((*bs, 4)),
                    lambda rng, bs=batch_shape: rng.standard_normal((*bs, 4)),  # loc
                    None,  # covariance_matrix
                    None,  # precision_matrix
                    lambda rng: numpy.linalg.cholesky(numpy.eye(4) + numpy.ones((4, 4))),
                    name="MultivariateNormal",
                    differentiable_argnums=(0, 1, 4),
                ),
                NumpyroDistributionTestConfig(
                    dists.LowRankMultivariateNormal,
                    lambda rng, bs=batch_shape: rng.standard_normal((*bs, 4)),  # loc
                    lambda rng, bs=batch_shape: rng.standard_normal((*bs, 4, 2)),  # cov_factor
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, (*bs, 4)),  # cov_diag
                    differentiable_argnums=(0, 1, 2, 3),
                ),
            ]
