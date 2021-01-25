# This script generates the model/fit used in some of the unit tests.

import biom
import pandas as pd
import pickle

from birdman.model import NegativeBinomial

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

nb.compile_model()
fit = nb.fit_model()

# need to pickle compiled model as well as fit
with open("tests/data/moving_pictures_model.pickle", "wb+") as f:
    pickle.dump({"model": nb.sm, "fit": fit}, f, pickle.HIGHEST_PROTOCOL)
