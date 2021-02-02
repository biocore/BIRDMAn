# This script generates the model/fit used in some of the unit tests.

import biom
import pandas as pd

from birdman import NegativeBinomial

table = biom.load_table("tests/data/table-deblur.biom")
metadata = pd.read_csv("tests/data/sample-metadata.tsv", sep="\t",
                       index_col=0)
metadata = metadata.loc[table.ids(axis="sample"), :]
metadata = metadata.rename(columns={"body-site": "body_site"})

nb = NegativeBinomial(
    table,
    "body_site",
    metadata,
    num_iter=500,
    seed=42,
)
fit = nb.fit_model()

ds = nb.to_xarray()
ds.to_netcdf("tests/data/moving_pictures_model.nc")
