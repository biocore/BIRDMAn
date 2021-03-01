import os
from pkg_resources import resource_filename

import pytest

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
    def test_nb_fit(self, table_biom, example_model):
        beta = example_model.fit.stan_variable("beta")
        phi = example_model.fit.stan_variable("phi")
        num_cov = 2
        num_chains = 4
        num_table_feats = 28
        num_iter = 100
        num_draws = num_chains*num_iter
        assert beta.shape == (num_draws, num_cov, num_table_feats - 1)
        assert phi.shape == (num_draws, num_table_feats)


class TestToInference:
    def test_serial_to_inference(self, example_model):
        inference_data = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"]
            },
            alr_params=["beta"],
            log_likelihood="log_lik",
            posterior_predictive="y_predict"
        )
        target_groups = {"posterior", "sample_stats", "log_likelihood",
                         "posterior_predictive"}
        assert set(inference_data.groups()) == target_groups

    def test_parallel_to_inference(self, example_parallel_model):
        inference_data = example_parallel_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_parallel_model.feature_names,
                "covariate": example_parallel_model.colnames
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"]
            },
        )
        target_groups = {"posterior", "sample_stats"}
        assert set(inference_data.groups()) == target_groups

    def test_parallel_to_inference_alr_to_clr(self, example_parallel_model):
        with pytest.warns(UserWarning) as w:
            inference_data = example_parallel_model.to_inference_object(
                params=["beta", "phi"],
                coords={
                    "feature": example_parallel_model.feature_names,
                    "covariate": example_parallel_model.colnames
                },
                dims={
                    "beta": ["covariate", "feature"],
                    "phi": ["feature"]
                },
                alr_params=["beta"]
            )

        assert w[0].message.args[0] == (
            "ALR to CLR not performed on parallel models."
        )
        target_groups = {"posterior", "sample_stats"}
        assert set(inference_data.groups()) == target_groups
