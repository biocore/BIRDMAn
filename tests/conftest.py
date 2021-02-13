import warnings

import biom
import pandas as pd
import pytest

from birdman import NegativeBinomial


def example_biom():
    table_biom = biom.load_table("tests/data/macaque_tbl.biom")
    return table_biom


def example_metadata():
    metadata = pd.read_csv(
        "tests/data/macaque_metadata.tsv",
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


@pytest.fixture(scope="session")
def example_model():
    tbl = example_biom()
    md = example_metadata()

    nb = NegativeBinomial(
        table=tbl,
        formula="host_common_name",
        metadata=md,
        num_iter=100,
        chains=4,
    )
    nb.compile_model()
    nb.fit_model()
    return nb


@pytest.fixture(scope="session")
def example_parallel_model():
    tbl = example_biom()
    md = example_metadata()

    nb = NegativeBinomial(
        table=tbl,
        formula="host_common_name",
        metadata=md,
        num_iter=100,
        chains=4,
        parallelize_across="features"
    )
    nb.compile_model()
    nb.fit_model()
    return nb
