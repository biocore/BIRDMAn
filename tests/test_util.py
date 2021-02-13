import numpy as np
from skbio.stats.composition import alr, clr

from birdman import util


def test_alr_to_clr():
    mat = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.3],
        [0.3, 0.1, 0.1, 0.2, 0.5],
        [0.4, 0.3, 0.5, 0.1, 0.1],
        [0.2, 0.4, 0.1, 0.3, 0.1]
    ])

    # skbio alr & clr take rows as compositions, columns as components
    alr_mat = alr(mat.T, 0)                 # 5 x 3
    clr_mat = util.alr_to_clr(alr_mat.T).T  # 5 x 4
    exp_clr = clr(mat.T)                    # 5 x 4

    np.testing.assert_array_almost_equal(clr_mat, exp_clr)


def test_convert_beta_coordinates():
    # Total: (n draws x p covariates x d features)
    # Each draw: (p covariates x d features)
    draw1 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.3, 0.1, 0.1, 0.5],
        [0.2, 0.2, 0.2, 0.3],
        [0.5, 0.1, 0.2, 0.2]
    ])
    draw2 = np.array([
        [0.2, 0.2, 0.3, 0.3],
        [0.1, 0.6, 0.2, 0.1],
        [0.4, 0.4, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7]
    ])
    # 4 covariates x (4-1) features x 2 draws
    alr_coords = np.dstack([alr(draw1), alr(draw2)])
    # 2 draws x 4 covariates x (4-1) features
    alr_coords = np.moveaxis(alr_coords, [2, 0, 1], [0, 1, 2])
    clr_coords = util.convert_beta_coordinates(alr_coords)  # p x (d+1) x n
    exp_coords = np.dstack([clr(draw1), clr(draw2)])
    np.testing.assert_array_almost_equal(clr_coords, exp_coords)

    clr_coords_sums = clr_coords.sum(axis=1)
    exp_clr_coords_sums = np.zeros((4, 2))
    np.testing.assert_array_almost_equal(exp_clr_coords_sums, clr_coords_sums)


class TestToXArray:
    def dataset_comparison(self, model, ds):
        coord_names = ds.coords._names
        assert coord_names == {"feature", "draw", "covariate"}
        assert ds["beta"].shape == (2, 28, 400)
        assert ds["phi"].shape == (28, 400)

        exp_feature_names = model.table.ids(axis="observation")
        ds_feature_names = ds.coords["feature"]
        assert (exp_feature_names == ds_feature_names).all()

        exp_coord_names = [
            "Intercept",
            "host_common_name[T.long-tailed macaque]"
        ]
        ds_coord_names = ds.coords["covariate"]
        assert (exp_coord_names == ds_coord_names).all()

    def test_serial_to_xarray(self, example_model):
        fit = example_model.fit
        ds = util.fit_to_xarray(
            fit=fit,
            params=["beta", "phi"],
            covariate_names=example_model.dmat.columns.tolist(),
            feature_names=example_model.table.ids(axis="observation")
        )
        self.dataset_comparison(example_model, ds)

    def test_parallel_to_xarray(self, example_parallel_model):
        fits = example_parallel_model.fits
        ds = util.fits_to_xarray(
            fits=fits,
            params=["beta", "phi"],
            covariate_names=example_parallel_model.dmat.columns.tolist(),
            feature_names=example_parallel_model.table.ids(axis="observation")
        )
        self.dataset_comparison(example_parallel_model, ds)
