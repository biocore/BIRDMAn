import numpy as np
import pytest

from birdman import model_util as mu


class TestToInference:
    def dataset_comparison(self, model, ds):
        coord_names = ds.coords._names
        assert coord_names == {"feature", "draw", "covariate", "chain"}
        assert set(ds["beta"].shape) == {2, 28, 4, 100}
        assert set(ds["phi"].shape) == {28, 4, 100}

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
        inf = mu.single_fit_to_inference(
            fit=example_model.fit,
            coords={
                "feature": example_model.feature_names,
                "covariate": example_model.colnames
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"]
            },
            params=["beta", "phi"],
            alr_params=["beta"]
        )
        self.dataset_comparison(example_model, inf.posterior)

    def test_parallel_to_inference(self, example_parallel_model):
        inf = mu.multiple_fits_to_inference(
            fits=example_parallel_model.fit,
            params=["beta", "phi"],
            coords={
                "feature": example_parallel_model.feature_names,
                "covariate": example_parallel_model.colnames
            },
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"]
            },
            concatenation_name="feature"
        )
        self.dataset_comparison(example_parallel_model, inf.posterior)

    def test_parallel_to_inference_wrong_concat(self, example_parallel_model):
        with pytest.raises(ValueError) as excinfo:
            mu.multiple_fits_to_inference(
                fits=example_parallel_model.fit,
                params=["beta", "phi"],
                coords={
                    "feature": example_parallel_model.feature_names,
                    "covariate": example_parallel_model.colnames
                },
                dims={
                    "beta": ["covariate", "feature"],
                    "phi": ["feature"]
                },
                concatenation_name="mewtwo"
            )
        assert str(excinfo.value) == ("concatenation_name must match "
                                      "dimensions in dims")
