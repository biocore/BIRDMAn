import os
from pkg_resources import resource_filename

from arviz import InferenceData
import numpy as np

from birdman import Multinomial, NegativeBinomial, NegativeBinomialLME

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

    def test_nb_lme(self, table_biom, metadata):
        md = metadata.copy()
        np.random.seed(42)
        md["group"] = np.random.randint(low=0, high=3, size=md.shape[0])
        md["group"] = "G" + md["group"].astype(str)
        nb_lme = NegativeBinomialLME(
            table=table_biom,
            formula="host_common_name",
            group_var="group",
            metadata=md,
            num_iter=100,
        )
        nb_lme.compile_model()
        nb_lme.fit_model()

        inf = nb_lme.to_inference_object()
        post = inf.posterior
        assert post["subj_int"].dims == ("chain", "draw", "group", "feature")
        assert post["subj_int"].shape == (4, 100, 3, 28)
        assert (post.coords["group"].values == ["G0", "G1", "G2"]).all()

    def test_mult(self, table_biom, metadata):
        md = metadata.copy()
        np.random.seed(42)
        mult = Multinomial(
            table=table_biom,
            formula="host_common_name",
            metadata=md,
            num_iter=100,
        )
        mult.compile_model()
        mult.fit_model()


class TestToInference:
    def test_serial_to_inference(self, example_model):
        inference_data = example_model.to_inference_object()
        target_groups = {"posterior", "sample_stats", "log_likelihood",
                         "posterior_predictive", "observed_data"}
        assert set(inference_data.groups()) == target_groups

    def test_parallel_to_inference(self, example_parallel_model):
        inference_data = example_parallel_model.to_inference_object()
        target_groups = {"posterior", "sample_stats", "log_likelihood",
                         "posterior_predictive", "observed_data"}
        assert set(inference_data.groups()) == target_groups

    def test_parallel_to_inference_no_concat(self, example_parallel_model):
        inf = example_parallel_model.to_inference_object(
            combine_individual_fits=False
        )
        assert len(inf) == 28

    def test_parallel_auto_inf(self, table_biom, metadata):
        nb = NegativeBinomial(
            table=table_biom,
            formula="host_common_name",
            metadata=metadata,
            chains=4,
            num_iter=100,
            seed=42,
            beta_prior=2.0,
            cauchy_scale=2.0,
            parallelize_across="features"
        )
        nb.compile_model()
        nb.fit_model(convert_to_inference=True)
        assert len(nb.fit) == 28
        assert isinstance(nb.fit[0], InferenceData)
