import os
from pkg_resources import resource_filename

import numpy as np
import pytest

from birdman import (NegativeBinomial, NegativeBinomialLME,
                     NegativeBinomialSingle, ModelIterator)

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
            inv_disp_sd=0.5,
        )

        assert nb.dat["B_p"] == 2.0
        assert nb.dat["inv_disp_sd"] == 0.5
        target_filepath = os.path.join(TEMPLATES, "negative_binomial.stan")
        assert nb.model_path == target_filepath


class TestModelFit:
    def test_nb_fit(self, table_biom, example_model):
        beta = example_model.fit.stan_variable("beta_var")
        inv_disp = example_model.fit.stan_variable("inv_disp")
        num_cov = 2
        num_chains = 4
        num_table_feats = 28
        num_iter = 100
        num_draws = num_chains*num_iter
        assert beta.shape == (num_draws, num_cov, num_table_feats - 1)
        assert inv_disp.shape == (num_draws, num_table_feats)

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

        inf = nb_lme.to_inference()
        post = inf.posterior
        assert post["subj_int"].dims == ("chain", "draw", "group",
                                         "feature_alr")
        assert post["subj_int"].shape == (4, 100, 3, 27)
        assert (post.coords["group"].values == ["G0", "G1", "G2"]).all()

    def test_single_feat(self, table_biom, metadata):
        md = metadata.copy()
        for fid in table_biom.ids(axis="observation"):
            nb = NegativeBinomialSingle(
                table=table_biom,
                feature_id=fid,
                formula="host_common_name",
                metadata=md,
                num_iter=100,
            )
            nb.compile_model()
            nb.fit_model(convert_to_inference=True)

    def test_fail_auto_conversion(self, table_biom, metadata):
        nb = NegativeBinomial(
            table=table_biom,
            metadata=metadata,
            formula="host_common_name",
            num_iter=100,
        )
        nb.compile_model()
        nb.specified = False
        with pytest.warns(UserWarning) as r:
            nb.fit_model(convert_to_inference=True)

        e = "Model has not been specified!"
        expected_warning = (
            "Auto conversion to InferenceData has failed! fit has "
            "been saved as CmdStanMCMC instead. See error message"
            f": \nValueError: {e}"
        )
        assert r[0].message.args[0] == expected_warning


class TestToInference:
    def test_serial_to_inference(self, example_model):
        inference_data = example_model.to_inference()
        target_groups = {"posterior", "sample_stats", "log_likelihood",
                         "posterior_predictive", "observed_data"}
        assert set(inference_data.groups()) == target_groups

    def test_single_feat_fit(self, example_single_feat_model):
        inf = example_single_feat_model.to_inference()
        post = inf.posterior
        assert set(post.coords) == {"chain", "covariate", "draw"}
        assert post.dims == {"chain": 4, "covariate": 2, "draw": 100}
        assert (post.coords["chain"] == [0, 1, 2, 3]).all()
        assert (post.coords["covariate"] == [
            "Intercept",
            "host_common_name[T.long-tailed macaque]"
        ]).all()
        assert (post.coords["draw"] == np.arange(100)).all()

        ppc = inf.posterior_predictive
        ll = inf.log_likelihood
        sample_names = example_single_feat_model.sample_names
        assert (ppc.coords["tbl_sample"] == sample_names).all()
        assert (ll.coords["tbl_sample"] == sample_names).all()


class TestModelIterator:
    def test_iteration(self, table_biom, metadata):
        model_iterator = ModelIterator(
            table=table_biom,
            model=NegativeBinomialSingle,
            formula="host_common_name",
            metadata=metadata,
            num_iter=100,
            chains=4,
            seed=42
        )

        iterated_feature_ids = []
        iterated_values = np.zeros(table_biom.shape)
        for i, (fid, model) in enumerate(model_iterator):
            iterated_feature_ids.append(fid)
            iterated_values[i] = model.dat["y"]

        expected_values = table_biom.to_dataframe(dense=True).values
        expected_feature_ids = table_biom.ids(axis="observation")

        np.testing.assert_equal(iterated_values, expected_values)
        assert (iterated_feature_ids == expected_feature_ids).all()

    def test_chunks(self, table_biom, metadata):
        chunk_size = 10

        model_iterator = ModelIterator(
            table=table_biom,
            model=NegativeBinomialSingle,
            formula="host_common_name",
            num_chunks=3,
            metadata=metadata,
            num_iter=100,
            chains=4,
            seed=42
        )

        tbl_fids = list(table_biom.ids("observation"))
        exp_chunk_fids = [
            tbl_fids[i: i+chunk_size]
            for i in range(0, table_biom.shape[0], chunk_size)
        ]

        chunk_sizes = []
        chunk_fids = []
        for i, chunk in enumerate(model_iterator):
            chunk_sizes.append(len(chunk))
            chunk_fids.append([x[0] for x in chunk])

        chunk_2 = model_iterator[1]

        assert chunk_sizes == [10, 10, 8]
        assert chunk_fids == exp_chunk_fids

        for (fid, _), exp_fid in zip(chunk_2, exp_chunk_fids[1]):
            assert fid == exp_fid

    def test_iteration_fit(self, table_biom, metadata):
        model_iterator = ModelIterator(
            table=table_biom,
            model=NegativeBinomialSingle,
            formula="host_common_name",
            metadata=metadata,
            num_iter=100,
            chains=4,
            seed=42
        )

        for fit, model in model_iterator:
            model.compile_model()
            model.fit_model()
            _ = model.to_inference()
