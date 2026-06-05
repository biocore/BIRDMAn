# BIRDMAn

[![GitHub Actions CI](https://github.com/biocore/birdman/workflows/BIRDMAn%20CI/badge.svg)](https://github.com/biocore/BIRDMAn/actions)
[![Documentation Status](https://readthedocs.org/projects/birdman/badge/?version=latest)](https://birdman.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/birdman.svg)](https://pypi.org/project/birdman)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4850650.svg)](https://doi.org/10.5281/zenodo.4850650)

## Overview

**B**ayesian **I**nferential **R**egression for **D**ifferential **M**icrobiome **An**alysis (**BIRDMAn**) is a framework for differential abundance analysis of microbiome count data using Bayesian inference.

Quantifying differential abundance (DA) in microbiome studies is challenging due to data characteristics (e.g., compositionality, sparsity) and study designs (e.g., repeated measures, meta-analysis, cross-over). BIRDMAn is a flexible DA method that accounts for these data characteristics and diverse experimental designs by fitting hierarchical Bayesian models in [Stan](https://mc-stan.org/) via [CmdStanPy](https://github.com/stan-dev/cmdstanpy), returning full posterior distributions over per-feature effects so that uncertainty is quantified alongside every estimate.

BIRDMAn ships with default negative-binomial models (fixed-effects and linear-mixed-effects variants, in both whole-table and per-feature forms) and exposes the underlying Stan templates so users can write their own custom models against the same fitting and post-processing machinery. Fits are returned as [ArviZ](https://python.arviz.org/) `InferenceData` objects.

See the [documentation](https://birdman.readthedocs.io/en/latest/?badge=latest) for details on usage, including a walkthrough of the default models and a guide to writing custom Stan code.

For a runnable example, see this [Google Colab notebook](https://colab.research.google.com/drive/1zT4eIgiz0Jl5TVmttE3gwWnrhNlPeYDc?usp=sharing).

## Installation

BIRDMAn requires Python 3.8 or higher.

We recommend using [mamba](https://github.com/mamba-org/mamba) for installation of dependencies.

```bash
mamba install -c conda-forge biom-format patsy xarray arviz cmdstanpy
pip install birdman
```

## Quick example

The snippet below uses the macaque gut microbiome table bundled with the repo at [`tests/data/`](tests/data) (clone the repo, or substitute your own BIOM table and metadata):

```python
import biom
import pandas as pd
from birdman import NegativeBinomial

table = biom.load_table("tests/data/macaque_tbl.biom")
metadata = pd.read_csv("tests/data/macaque_metadata.tsv", sep="\t", index_col=0)
metadata.index = metadata.index.astype(str)  # match BIOM sample IDs

nb = NegativeBinomial(
    table=table,
    formula="host_common_name",
    metadata=metadata,
)
nb.compile_model()
nb.fit_model(method="vi", num_draws=500)

inference = nb.to_inference()
inference.posterior["beta_var"]  # per-covariate, per-feature effects
```

## Citation

If you use BIRDMAn, please cite our manuscript:

> Rahman G, Morton JT, Martino C, Sepich-Poore GD, Allaband C, Guccione C, Chen Y, Hakim D, Estaki M, Knight R. BIRDMAn: A Bayesian differential abundance framework that enables robust inference of host-microbe associations. bioRxiv. 2023. doi:[10.1101/2023.01.30.526328](https://doi.org/10.1101/2023.01.30.526328).

## Contributing

Issues and pull requests are welcome.

## License

BIRDMAn is released under the BSD 3-Clause License, see [`LICENSE`](LICENSE).
