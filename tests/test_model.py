import os
from pkg_resources import resource_filename

from birdman import NegativeBinomial

TEMPLATES = resource_filename("birdman", "templates")


class TestModelInheritance:
    def test_nb_model(self, table_biom, metadata):
        nb = NegativeBinomial(
            table=table_biom,
            formula="host_common_name",
            metadata=metadata,
            chains=4,
            seed=42,
            beta_prior=2.0,
            cauchy_scale=2.0,
            parallelize_across="chains"
        )

        assert nb.dat["B_p"] == 2.0
        assert nb.dat["phi_s"] == 2.0
        target_filepath = os.path.join(TEMPLATES, "negative_binomial.stan")
        assert nb.model_path == target_filepath


class TestModelFit:
    def test_nb_fit(self, table_biom, example_fit):
        beta = example_fit.fit.stan_variable("beta")
        phi = example_fit.fit.stan_variable("phi")
        num_cov = 2
        num_chains = 4
        num_table_feats = 28
        num_iter = 100
        num_draws = num_chains*num_iter
        assert beta.shape == (num_draws, num_cov, num_table_feats - 1)
        assert phi.shape == (num_draws, num_table_feats)
