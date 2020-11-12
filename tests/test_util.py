import numpy as np
from skbio.stats.composition import alr, clr

from songbird2 import util


def test_collapse_results(exp_model):
    exp_res = exp_model.extract(permuted=True)
    num_covariates = exp_res["beta"].shape[1]
    colnames = [f"C{i+0}" for i in range(num_covariates)]
    beta_df = util.collapse_results(exp_res["beta"], colnames)

    assert beta_df.shape == (29, 8)
    assert not beta_df.isna().any().any()


def test_alr_to_clr():
    mat = np.array([
        [0.5, 0.2, 0.3],
        [0.4, 0.1, 0.4],
        [0.1, 0.1, 0.8],
        [0.3, 0.4, 0.3],
    ])

    alr_mat = alr(mat, 0)
    clr_mat = util.alr_to_clr(alr_mat)
    exp_clr = clr(mat)

    np.testing.assert_array_almost_equal(clr_mat, exp_clr)
