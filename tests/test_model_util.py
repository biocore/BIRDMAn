import numpy as np
from skbio.stats import composition as comp

from birdman.model import Model
import birdman.model_util as mutil
from birdman.util import alr_to_clr


def test_collapse_matrix(ex_model, table_biom, metadata, mocker):
    model = Model(table_biom, "body_site", metadata, model_type="test")
    model.colnames = ["x", "y", "z"]
    probs_1 = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.1, 0.6],
        [0.5, 0.2, 0.3],
    ])
    probs_2 = np.array([
        [0.15, 0.25, 0.6],
        [0.33, 0.44, 0.23],
        [0.05, 0.85, 0.10],

    ])
    alr_1 = comp.alr(probs_1)
    alr_2 = comp.alr(probs_2)
    mock_res = {"beta": np.stack([alr_1, alr_2], axis=0)}
    mocker.patch(  # mock to use test ALR coordinates for collapsing
        "birdman.model_util._extract_params",
        return_value=mock_res,
    )

    # calculate mean & std of all 3 vbls
    clr_1 = alr_to_clr(alr_1)
    clr_2 = alr_to_clr(alr_2)
    target_mean = np.zeros([3, 3])
    target_std = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            x = clr_1[i, j]
            y = clr_2[i, j]
            # [j, i] because we want each vbl to be one column
            target_mean[j, i] = np.mean([x, y])
            target_std[j, i] = np.std([x, y])

    param_df = mutil.collapse_param(model, "beta", convert_alr_to_clr=True)
    target_colnames = ["x_mean", "x_std", "y_mean", "y_std", "z_mean", "z_std"]

    assert param_df.columns.tolist() == target_colnames
    for i in range(3):
        np.testing.assert_allclose(  # test means
            param_df.iloc[:, 2*i].values, target_mean[:, i]
        )
        np.testing.assert_allclose(  # test stds
            param_df.iloc[:, 2*i+1].values, target_std[:, i]
        )


def test_collapse_vector(ex_model, table_biom, metadata, mocker):
    model = Model(table_biom, "body_site", metadata, model_type="test")
    vals_1 = np.array([1, 2, 3, 4, 5])
    vals_2 = np.array([2, 4, 6, 8, 10])
    mock_res = {"phi": np.stack([vals_1, vals_2], axis=0)}
    mocker.patch(  # mock to use test phi values for collapsing
        "birdman.model_util._extract_params",
        return_value=mock_res
    )

    target_mean = np.array([1.5, 3, 4.5, 6, 7.5])
    target_std = np.array([0.5, 1, 1.5, 2, 2.5])

    param_df = mutil.collapse_param(model, "phi", convert_alr_to_clr=False)

    assert param_df.columns.to_list() == ["phi_mean", "phi_std"]
    np.testing.assert_allclose(
        param_df["phi_mean"].to_numpy(), target_mean
    )
    np.testing.assert_allclose(
        param_df["phi_std"].to_numpy(), target_std
    )
