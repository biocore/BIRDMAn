import biom
import numpy as np
import pandas as pd
import pickle
import pytest

from qiime2 import Artifact


@pytest.fixture
def data_table():
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

    table_biom = biom.table.Table(
        table_filt.T.values,
        random_features,
        random_samples,
    )
    return table_biom


@pytest.fixture
def exp_res():
    with open("tests/data/stan_res.npy", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def metadata():
    metadata = pd.read_csv(
        "tests/data/sample-metadata.tsv",
        sep="\t",
        index_col=0,
    )
    metadata = metadata.rename(columns={"body-site": "body_site"})
    return metadata
