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
