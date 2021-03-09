import pytest

from birdman import visualization as viz


class TestRankPlot:
    def test_rank_plot_beta(self, example_model):
        inference = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames,
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
            },
            alr_params=["beta"],
        )

        viz.plot_parameter_estimates(
            inference_object=inference,
            parameter="beta",
            coord={"covariate": "host_common_name[T.long-tailed macaque]"},
            num_std=1
        )

    def test_rank_plot_phi(self, example_model):
        inference = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames,
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
            },
            alr_params=["beta"],
        )

        viz.plot_parameter_estimates(
            inference_object=inference,
            parameter="phi",
            num_std=1
        )

    def test_rank_plot_no_coord(self, example_model):
        inference = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames,
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
            },
            alr_params=["beta"],
        )

        with pytest.raises(ValueError) as excinfo:
            viz.plot_parameter_estimates(
                inference_object=inference,
                parameter="beta",
                num_std=1
            )

        exp_msg = ("Must provide coordinates if plotting multi-dimensional "
                   "parameter estimates!")
        assert str(excinfo.value) == exp_msg


class TestPPCPlot:
    def test_ppc(self, example_model):
        inference = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames,
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
            },
            alr_params=["beta"],
            include_observed_data=True,
            posterior_predictive="y_predict"
        )

        viz.plot_posterior_predictive_checks(inference)

    def test_ppc_no_pp(self, example_model):
        inference = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames,
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
            },
            alr_params=["beta"],
            include_observed_data=True,
        )

        with pytest.raises(ValueError) as excinfo:
            viz.plot_posterior_predictive_checks(inference)

        exp_msg = "Must include posterior predictive values to perform PPC!"
        assert str(excinfo.value) == exp_msg

    def test_ppc_no_obs(self, example_model):
        inference = example_model.to_inference_object(
            params=["beta", "phi"],
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames,
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
            },
            alr_params=["beta"],
            posterior_predictive="y_predict"
        )

        with pytest.raises(ValueError) as excinfo:
            viz.plot_posterior_predictive_checks(inference)

        exp_msg = "Must include observed data to perform PPC!"
        assert str(excinfo.value) == exp_msg
