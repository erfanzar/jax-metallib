import numpy
from jax import random
from numpyro import distributions as dists

from .util import OperationTestConfig


class NumpyroDistributionTestConfig(OperationTestConfig):
    """
    Test config for numpyro distributions that evaluates log_prob and samples.

    Args:
        dist_cls: Distribution class to test.
        *args: Positional arguments passed to the distribution constructor.
        **kwargs: Keyword arguments passed to OperationTestConfig.
    """

    def __init__(self, dist_cls: type[dists.Distribution], *args, **kwargs) -> None:
        kwargs.setdefault("name", dist_cls.__name__)
        super().__init__(
            lambda x, *args: dist_cls(*args).log_prob(x).mean(),
            lambda rng: dist_cls(*[a(rng) if callable(a) else a for a in args]).sample(
                random.key(17)
            ),
            *args,
            **kwargs,
        )


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
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs + (3,)),
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
                    lambda rng, bs=batch_shape: rng.uniform(
                        -numpy.pi, numpy.pi, bs
                    ),  # loc
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                ),
                # Multivariate distributions.
                OperationTestConfig(
                    lambda x, loc, covariance_matrix, precision_matrix, scale_tril: dists.MultivariateNormal(
                        loc, covariance_matrix, precision_matrix, scale_tril
                    )
                    .log_prob(x)
                    .mean(),
                    lambda rng, bs=batch_shape: rng.standard_normal(bs + (4,)),
                    lambda rng, bs=batch_shape: rng.standard_normal(bs + (4,)),  # loc
                    None,  # covariance_matrix
                    None,  # precision_matrix
                    lambda rng: numpy.linalg.cholesky(numpy.eye(4) + numpy.ones((4, 4))),
                    name="MultivariateNormal",
                    differentiable_argnums=(0, 1, 4),
                ),
                NumpyroDistributionTestConfig(
                    dists.LowRankMultivariateNormal,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs + (4,)),  # loc
                    lambda rng, bs=batch_shape: rng.standard_normal(bs + (4, 2)),  # cov_factor
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs + (4,)),  # cov_diag
                    differentiable_argnums=(0, 1, 2, 3),
                ),
            ]
