import os
from pkg_resources import resource_filename

import numpy as np
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
    # This is the same as in test_model_util
    # TODO: Create auxillary file to hold this and other utility functions
    def dataset_comparison(self, model, ds):
        coord_names = ds.coords._names
        assert coord_names == {"feature", "draw", "covariate", "chain"}
        assert ds["beta"].shape == (2, 28, 4, 100)
        assert ds["phi"].shape == (28, 4, 100)

        exp_feature_names = model.table.ids(axis="observation")
        ds_feature_names = ds.coords["feature"]
        assert (exp_feature_names == ds_feature_names).all()

        exp_coord_names = [
            "Intercept",
            "host_common_name[T.long-tailed macaque]"
        ]
        ds_coord_names = ds.coords["covariate"]
        assert (exp_coord_names == ds_coord_names).all()

        assert (ds.coords["draw"] == np.arange(100)).all()
        assert (ds.coords["chain"] == [0, 1, 2, 3]).all()

    def test_serial_to_inference(self, example_model):
        inference_data = example_model.to_inference_object(
            params_to_include=["beta", "phi"],
            alr_params=["beta"],
        )
        assert inference_data.groups() == ["posterior"]
        self.dataset_comparison(example_model, inference_data.posterior)

    def test_parallel_to_inference(self, example_parallel_model):
        inference_data = example_parallel_model.to_inference_object(
            params_to_include=["beta", "phi"],
        )
        assert inference_data.groups() == ["posterior"]
        self.dataset_comparison(example_parallel_model,
                                inference_data.posterior)

    def test_parallel_to_inference_alr_to_clr(self, example_parallel_model):
        with pytest.warns(UserWarning) as w:
            inference_data = example_parallel_model.to_inference_object(
                params_to_include=["beta", "phi"],
                alr_params=["beta"],
            )

        assert w[0].message.args[0] == (
            "ALR to CLR not performed on parallel models."
        )
        assert inference_data.groups() == ["posterior"]
        self.dataset_comparison(example_parallel_model,
                                inference_data.posterior)
