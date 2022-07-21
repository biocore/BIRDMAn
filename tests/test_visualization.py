import pytest

from birdman import visualization as viz


class TestRankPlot:
    def test_rank_plot_beta(self, example_inf):
        viz.plot_parameter_estimates(
            inference_object=example_inf,
            parameter="beta_var",
            coords={"covariate": "host_common_name[T.long-tailed macaque]"},
            num_std=1
        )

    def test_rank_plot_inv_disp(self, example_inf):
        viz.plot_parameter_estimates(
            inference_object=example_inf,
            parameter="inv_disp",
            num_std=1
        )

    def test_rank_plot_no_coords(self, example_inf):
        with pytest.raises(ValueError) as excinfo:
            viz.plot_parameter_estimates(
                inference_object=example_inf,
                parameter="beta_var",
                num_std=1
            )

        exp_msg = ("Must provide coordinates if plotting multi-dimensional "
                   "parameter estimates!")
        assert str(excinfo.value) == exp_msg


class TestPPCPlot:
    def test_ppc(self, example_inf):
        viz.plot_posterior_predictive_checks(example_inf)

    def test_ppc_no_pp(self, example_model):
        inference = example_model.to_inference().copy()
        delattr(inference, "posterior_predictive")

        with pytest.raises(ValueError) as excinfo:
            viz.plot_posterior_predictive_checks(inference)

        exp_msg = "Must include posterior predictive values to perform PPC!"
        assert str(excinfo.value) == exp_msg

    def test_ppc_no_obs(self, example_model):
        inference = example_model.to_inference().copy()
        delattr(inference, "observed_data")

        with pytest.raises(ValueError) as excinfo:
            viz.plot_posterior_predictive_checks(inference)

        exp_msg = "Must include observed data to perform PPC!"
        assert str(excinfo.value) == exp_msg
