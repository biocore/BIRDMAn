import numpy as np

from birdman import model_util as mu


class TestToXArray:
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

    def test_serial_to_xarray(self, example_model):
        ds = mu.single_fit_to_xarray(
            fit=example_model.fit,
            params=["beta", "phi"],
            covariate_names=example_model.dmat.columns.tolist(),
            feature_names=example_model.table.ids(axis="observation")
        )
        self.dataset_comparison(example_model, ds)

    def test_parallel_to_xarray(self, example_parallel_model):
        ds = mu.multiple_fits_to_xarray(
            fits=example_parallel_model.fit,
            params=["beta", "phi"],
            covariate_names=example_parallel_model.dmat.columns.tolist(),
            feature_names=example_parallel_model.table.ids(axis="observation")
        )
        self.dataset_comparison(example_parallel_model, ds)
