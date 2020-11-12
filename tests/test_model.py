import numpy as np
from patsy import dmatrix

from songbird2 import fit_model
from songbird2.model import _fit


class TestModel:
    def test_fit_model(self, mocker, data_table, metadata, exp_model):
        mocker.patch("songbird2.model._fit", return_value=exp_model)
        fit_model(
            data_table,
            metadata,
            "body_site",
            num_iter=500,
            seed=42,
        )

    def test_fit(self, data_table, metadata, exp_model):
        _, exp_fit = exp_model
        exp_res = exp_fit.extract(permuted=True)

        table_df = data_table.to_dataframe().T
        metadata_filt = metadata.loc[table_df.index, :]
        dmat = dmatrix("body_site", metadata_filt, return_type="dataframe")

        _, fit = _fit(
            table_df.values,
            dmat,
            num_iter=500,
            seed=42,
        )
        res = fit.extract(permuted=True)

        np.testing.assert_almost_equal(exp_res["beta"],  res["beta"])
