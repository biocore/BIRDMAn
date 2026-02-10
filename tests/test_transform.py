import numpy as np
import xarray as xr
from skbio.stats.composition import alr, clr

from birdman import transform


def test_posterior_alr_to_clr(example_model):
    inf = example_model.to_inference()
    np.testing.assert_equal(
        inf.posterior["beta_var"].coords["feature_alr"],
        example_model.feature_names[1:],
    )

    new_post = transform.posterior_alr_to_clr(
        inf.posterior,
        alr_params=["beta_var"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=example_model.feature_names
    )

    assert set(new_post.dims) == {"chain", "draw", "feature", "covariate"}
    np.testing.assert_equal(
        new_post.coords["feature"],
        example_model.feature_names
    )


def test_alr_to_clr():
    mat = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.3],
        [0.3, 0.1, 0.1, 0.2, 0.5],
        [0.4, 0.3, 0.5, 0.1, 0.1],
        [0.2, 0.4, 0.1, 0.3, 0.1]
    ])

    # skbio alr & clr take rows as compositions, columns as components
    alr_mat = alr(mat.T, 0)                 # 5 x 3
    clr_mat = transform._alr_to_clr(alr_mat.T).T  # 5 x 4
    exp_clr = clr(mat.T)                    # 5 x 4

    np.testing.assert_array_almost_equal(clr_mat, exp_clr)


def test_clr_to_alr():
    mat = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.3],
        [0.3, 0.1, 0.1, 0.2, 0.5],
        [0.4, 0.3, 0.5, 0.1, 0.1],
        [0.2, 0.4, 0.1, 0.3, 0.1]
    ])

    # skbio alr & clr take rows as compositions, columns as components
    clr_mat = clr(mat.T)
    alr_mat = transform._clr_to_alr(clr_mat.T).T
    exp_alr = alr(mat.T)

    np.testing.assert_array_almost_equal(alr_mat, exp_alr)


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
    alr_coords = np.stack([alr(draw1), alr(draw2)])  # 2 x 4 x 3
    clr_coords = transform._beta_alr_to_clr(alr_coords)  # 2 x 4 x 4
    exp_coords = np.stack([clr(draw1), clr(draw2)])
    np.testing.assert_array_almost_equal(clr_coords, exp_coords)

    clr_coords_sums = clr_coords.sum(axis=2)
    exp_clr_coords_sums = np.zeros((2, 4))
    np.testing.assert_array_almost_equal(exp_clr_coords_sums, clr_coords_sums)


def test_posterior_alr_to_clr_multi_chain():
    """Regression test: groupby chain dimension not squeezed."""
    num_chains, num_draws, num_covariates, num_features_alr = 4, 50, 3, 5
    num_features = num_features_alr + 1

    beta_data = np.random.randn(
        num_chains, num_draws, num_covariates, num_features_alr
    )
    feature_names = [f"feat{i}" for i in range(num_features)]
    feature_names_alr = feature_names[1:]
    covariate_names = [f"cov{i}" for i in range(num_covariates)]

    ds = xr.Dataset({
        "beta_var": xr.DataArray(
            beta_data,
            dims=["chain", "draw", "covariate", "feature_alr"],
            coords={
                "chain": np.arange(num_chains),
                "draw": np.arange(num_draws),
                "covariate": covariate_names,
                "feature_alr": feature_names_alr,
            },
        )
    })

    result = transform.posterior_alr_to_clr(
        ds,
        alr_params=["beta_var"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=feature_names,
    )

    assert set(result.dims) == {"chain", "draw", "feature", "covariate"}
    assert result["beta_var"].shape == (
        num_chains, num_draws, num_covariates, num_features
    )
    np.testing.assert_equal(result.coords["feature"].values, feature_names)
    assert np.allclose(result["beta_var"].values.sum(axis=-1), 0, atol=1e-10)


def test_posterior_alr_to_clr_single_chain_single_covariate():
    """Verify fix works with VI (1 chain) and intercept-only (1 covariate)."""
    num_chains, num_draws, num_covariates, num_features_alr = 1, 50, 1, 5
    num_features = num_features_alr + 1

    beta_data = np.random.randn(
        num_chains, num_draws, num_covariates, num_features_alr
    )
    feature_names = [f"feat{i}" for i in range(num_features)]
    feature_names_alr = feature_names[1:]
    covariate_names = ["Intercept"]

    ds = xr.Dataset({
        "beta_var": xr.DataArray(
            beta_data,
            dims=["chain", "draw", "covariate", "feature_alr"],
            coords={
                "chain": np.arange(num_chains),
                "draw": np.arange(num_draws),
                "covariate": covariate_names,
                "feature_alr": feature_names_alr,
            },
        )
    })

    result = transform.posterior_alr_to_clr(
        ds,
        alr_params=["beta_var"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=feature_names,
    )

    assert result["beta_var"].shape == (
        num_chains, num_draws, num_covariates, num_features
    )
    assert np.allclose(result["beta_var"].values.sum(axis=-1), 0, atol=1e-10)


def test_posterior_alr_to_clr_multiple_params():
    """Verify multiple alr_params (e.g. beta_var + subj_int) all transform."""
    num_chains, num_draws = 2, 50
    num_covariates, num_groups = 3, 4
    num_features_alr, num_features = 5, 6

    feature_names = [f"feat{i}" for i in range(num_features)]
    feature_names_alr = feature_names[1:]

    ds = xr.Dataset({
        "beta_var": xr.DataArray(
            np.random.randn(
                num_chains, num_draws, num_covariates, num_features_alr
            ),
            dims=["chain", "draw", "covariate", "feature_alr"],
            coords={
                "chain": np.arange(num_chains),
                "draw": np.arange(num_draws),
                "covariate": [f"cov{i}" for i in range(num_covariates)],
                "feature_alr": feature_names_alr,
            },
        ),
        "subj_int": xr.DataArray(
            np.random.randn(
                num_chains, num_draws, num_groups, num_features_alr
            ),
            dims=["chain", "draw", "group", "feature_alr"],
            coords={
                "chain": np.arange(num_chains),
                "draw": np.arange(num_draws),
                "group": [f"subj{i}" for i in range(num_groups)],
                "feature_alr": feature_names_alr,
            },
        ),
    })

    result = transform.posterior_alr_to_clr(
        ds,
        alr_params=["beta_var", "subj_int"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=feature_names,
    )

    assert result["beta_var"].shape == (
        num_chains, num_draws, num_covariates, num_features
    )
    assert result["subj_int"].shape == (
        num_chains, num_draws, num_groups, num_features
    )
    assert np.allclose(result["beta_var"].values.sum(axis=-1), 0, atol=1e-10)
    assert np.allclose(result["subj_int"].values.sum(axis=-1), 0, atol=1e-10)
