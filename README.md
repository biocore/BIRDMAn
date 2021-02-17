# BIRDMAn

[![GitHub Actions CI](https://github.com/gibsramen/birdman/workflows/BIRDMAn%20CI/badge.svg)](https://github.com/gibsramen/BIRDMAn/actions)
[![Documentation Status](https://readthedocs.org/projects/birdman/badge/?version=latest)](https://birdman.readthedocs.io/en/latest/?badge=latest)

**B**ayesian **I**nferential **R**egression for **D**ifferential **M**icrobiome **An**alysis (**BIRDMAn**) is a framework for performing differential abundance analysis through Bayesian inference.

Much of the code in this repository has been adapted from and inspired by [Jamie Morton](https://mortonjt.github.io/probable-bug-bytes/probable-bug-bytes/differential-abundance/)

## Installation

```bash
conda install -c conda-forge dask biom-format patsy xarray
pip install cmdstanpy
git clone git@github.com:gibsramen/BIRDMAn.git
cd BIRDMAn
pip install .
```

You have to install `cmdstan` through the `cmdstanpy.install_cmdstan` function first.

## Example Usage

```python
import biom
from birdman import NegativeBinomial
import pandas as pd

tbl = biom.load_table(“table.biom”)
metadata = pd.read_csv(“metadata.tsv”, sep=“\t”)

nb = NegativeBinomial(
    table=tbl,
    formula=“ph+diet”,
    metadata=metadata,
    num_iter=1000,
    beta_prior=2.0,
    cauchy_scale=5.0
)

nb.compile_model()
nb.fit_model()

fit = nb.fit
# done
```
