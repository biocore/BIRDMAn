import numpy as np

from birdman import model_util as mu


class TestToInference:
    def dataset_comparison(self, model, ds):
        coord_names = ds.coords._names
        exp_coords = {"feature", "draw", "covariate", "chain", "feature_alr"}
        assert exp_coords.issubset(coord_names)
        assert set(ds["beta"].shape) == {2, 27, 4, 100}
        assert set(ds["phi"].shape) == {28, 4, 100}

        exp_feature_names = model.feature_names
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
        inf = mu.full_fit_to_inference(
            fit=example_model.fit,
            coords={
                "feature": example_model.feature_names,
                "feature_alr": example_model.feature_names[1:],
                "covariate": example_model.colnames,
                "tbl_sample": example_model.sample_names
            },
            dims={
                "beta": ["covariate", "feature_alr"],
                "phi": ["feature"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            params=["beta", "phi"],
            alr_params=["beta"]
        )
        self.dataset_comparison(example_model, inf.posterior)


# Posterior predictive & log likelihood
class TestPPLL:
    def test_serial_ppll(self, example_model):
        inf = mu.full_fit_to_inference(
            fit=example_model.fit,
            coords={
                "feature": example_model.feature_names,
                "feature_alr": example_model.feature_names[1:],
                "covariate": example_model.colnames,
                "sample": example_model.sample_names,
            },
            dims={
                "beta": ["covariate", "feature_alr"],
                "phi": ["feature"],
                "log_lhood": ["sample", "feature"],
                "y_predict": ["sample", "feature"]
            },
            params=["beta", "phi"],
            alr_params=["beta"],
            posterior_predictive="y_predict",
            log_likelihood="log_lhood",
        )

        d = {"posterior_predictive": "y_predict",
             "log_likelihood": "log_lhood"}
        for k, v in d.items():
            inf_data = inf[k][v].values
            nb_data = example_model.fit.stan_variable(v)
            nb_data = np.array(np.split(nb_data, 4, axis=0))
            np.testing.assert_array_almost_equal(nb_data, inf_data)
