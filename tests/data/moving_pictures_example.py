# This script generates the model/fit used in some of the unit tests.

import numpy as np
import pandas as pd
from patsy import dmatrix
import pickle
from qiime2 import Artifact

from birdman.model import NegativeBinomial

table = Artifact.load("tests/data/table-deblur.qza")
table_df = table.view(pd.DataFrame)

np.random.seed(42)
random_samples = table_df.sample(20).index
random_features = table_df.T.sample(30).index
table_filt = table_df.loc[random_samples, random_features]

sample_sums = table_filt.sum(axis=1)
feature_sums = table_filt.sum(axis=0)
random_samples = sample_sums[sample_sums > 0].index.tolist()
random_features = feature_sums[feature_sums > 0].index.tolist()

table_filt = table_filt.loc[random_samples, :]
table_filt = table_filt.loc[:, random_features]

metadata = pd.read_csv("tests/data/sample-metadata.tsv", sep="\t",
                       index_col=0)
metadata = metadata.loc[table_filt.index, :]
metadata = metadata.rename(columns={"body-site": "body_site"})

dmat = dmatrix("body_site", metadata, return_type="dataframe")

nb = NegativeBinomial(
    table_filt,
    dmat,
    num_iter=500,
    seed=42,
)

nb.compile_model()
fit = nb.fit_model()

# need to pickle compiled model as well as fit
with open("tests/data/moving_pictures_model.pickle", "wb+") as f:
    pickle.dump({"model": nb.sm, "fit": fit}, f, pickle.HIGHEST_PROTOCOL)
