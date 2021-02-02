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
