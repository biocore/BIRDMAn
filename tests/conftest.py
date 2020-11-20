import biom
import numpy as np
import pandas as pd
import pickle
import pytest

from patsy import dmatrix
from qiime2 import Artifact


def example_table():
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

    return table_filt


def example_metadata():
    metadata = pd.read_csv(
        "tests/data/sample-metadata.tsv",
        sep="\t",
        index_col=0,
    )
    metadata = metadata.rename(columns={"body-site": "body_site"})
    return metadata


@pytest.fixture
def table_biom():
    table_filt = example_table()

    table_biom = biom.table.Table(
        table_filt.T.values,
        table_filt.columns,
        table_filt.samples,
    )
    return table_biom


@pytest.fixture
def table_df():
    return example_table()


@pytest.fixture
def metadata():
    return example_metadata()


@pytest.fixture
def dmat():
    md = example_metadata()
    tbl = example_table()
    md_filt = md.loc[tbl.index, :]
    dmat = dmatrix("body_site", md_filt, return_type="dataframe")
    return dmat


@pytest.fixture
def ex_model():
    with open("tests/data/moving_pictures_model.pickle", "rb") as f:
        load_fit = pickle.load(f)
    return load_fit
