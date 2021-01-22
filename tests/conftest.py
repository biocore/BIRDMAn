import biom
import pandas as pd
import pickle
import pytest

from patsy import dmatrix


def example_table():
    table_biom = biom.load_table("tests/data/table-deblur.biom")
    return table_biom.to_dataframe().T


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
    table_biom = biom.load_table("tests/data/table-deblur.biom")
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
