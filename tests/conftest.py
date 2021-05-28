import os
from pkg_resources import resource_filename

import biom
import pandas as pd
import pytest

from birdman import NegativeBinomial

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
        num_iter=100,
        chains=4,
    )
    nb.compile_model()
    nb.fit_model()
    return nb


@pytest.fixture(scope="session")
def example_model():
    return model()


def parallel_model():
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


@pytest.fixture(scope="session")
def example_parallel_model():
    return parallel_model()


@pytest.fixture(scope="session")
def example_inf():
    nb = model()
    inference = nb.to_inference_object()
    return inference


@pytest.fixture(scope="session")
def example_parallel_inf():
    nb = parallel_model()
    inference = nb.to_inference_object(combine_individual_fits=True)
    return inference
