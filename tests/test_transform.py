import numpy as np
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
