import numpy as np
from patsy import dmatrix

from songbird2.model import NegativeBinomial


class TestModel:
    def test_fit(self, data_table, metadata, exp_model):
        table_df = data_table.to_dataframe().T
        metadata_filt = metadata.loc[table_df.index, :]
        dmat = dmatrix("body_site", metadata_filt, return_type="dataframe")

        nb = NegativeBinomial(
            table_df,
            dmat,
            num_iter=500,
            seed=42,
        )

        nb._fit()
