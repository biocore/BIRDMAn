# BIRDMAn

[![GitHub Actions CI](https://github.com/biocore/birdman/workflows/BIRDMAn%20CI/badge.svg)](https://github.com/biocore/BIRDMAn/actions)
[![Documentation Status](https://readthedocs.org/projects/birdman/badge/?version=stable)](https://birdman.readthedocs.io/en/stable/?badge=stable)
[![PyPI](https://img.shields.io/pypi/v/birdman.svg)](https://pypi.org/project/birdman)
[![DOI](https://zenodo.org/badge/312046610.svg)](https://zenodo.org/badge/latestdoi/312046610)

**B**ayesian **I**nferential **R**egression for **D**ifferential **M**icrobiome **An**alysis (**BIRDMAn**) is a framework for performing differential abundance analysis through Bayesian inference.

See the [documentation](https://birdman.readthedocs.io/en/stable/?badge=stable) for details on usage.

For an example of running BIRDMAn - see this [Google Colab notebook](https://colab.research.google.com/drive/1zT4eIgiz0Jl5TVmttE3gwWnrhNlPeYDc?usp=sharing).

## Installation

Currently BIRDMAn requires Python 3.8 or higher.

We recommend using [mamba](https://github.com/mamba-org/mamba) for installation of dependencies.

```bash
mamba install -c conda-forge biom-format patsy xarray arviz cmdstanpy
pip install birdman
```
