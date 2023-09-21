import os
from pkg_resources import resource_filename

import biom
import pandas as pd
import pytest

from birdman import NegativeBinomial, NegativeBinomialSingle

TEST_DATA = resource_filename("tests", "data")
TBL_FILE = os.path.join(TEST_DATA, "macaque_tbl.biom")
MD_FILE = os.path.join(TEST_DATA, "macaque_metadata.tsv")


def example_biom():
    table_biom = biom.load_table(TBL_FILE)
    return table_biom


def example_metadata():
    metadata = pd.read_csv(
        MD_FILE,
        sep="\t",
        index_col=0,
    )
    metadata.index = metadata.index.astype(str)
    return metadata


@pytest.fixture
def table_biom():
    return example_biom()


@pytest.fixture
def metadata():
    return example_metadata()


def model():
    tbl = example_biom()
    md = example_metadata()

    nb = NegativeBinomial(
        table=tbl,
        formula="host_common_name",
        metadata=md,
    )
    nb.compile_model()
    nb.fit_model(method="mcmc", mcmc_chains=4, num_draws=100)
    return nb


@pytest.fixture(scope="session")
def example_model():
    return model()


@pytest.fixture(scope="session")
def example_inf():
    nb = model()
    inference = nb.to_inference()
    return inference


def single_feat_model():
    tbl = example_biom()
    md = example_metadata()

    id0 = tbl.ids(axis="observation")[0]
    nb = NegativeBinomialSingle(
        table=tbl,
        formula="host_common_name",
        metadata=md,
        feature_id=id0,
    )

    nb.compile_model()
    nb.fit_model(method="mcmc", mcmc_chains=4, num_draws=100)

    return nb


@pytest.fixture(scope="session")
def example_single_feat_model():
    return single_feat_model()
