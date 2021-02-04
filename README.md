# BIRDMAn

[![GitHub Actions CI](https://github.com/gibsramen/birdman/workflows/BIRDMAn%20CI/badge.svg)](https://github.com/gibsramen/BIRDMAn/actions)

**B**ayesian **I**nferential **R**egression for **D**ifferential **M**icrobiome **An**alysis (**BIRDMAn**) is a framework for performing differential abundance analysis through Bayesian inference.

Much of the code in this repository has been adapted from and inspired by [Jamie Morton](https://mortonjt.github.io/probable-bug-bytes/probable-bug-bytes/differential-abundance/)

## Installation

BIRDMAn requires at at least Python 3.7 but has not been tested on newer versions. It is recommended to create a new conda environment for running BIRDMAn.

```bash
pip install pystan==3.0.0b7
conda install -c conda-forge dask biom-format patsy xarray
git clone git@github.com:gibsramen/BIRDMAn.git
cd BIRDMAn
pip install .
```

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

fit = nb.fit_model()
```
