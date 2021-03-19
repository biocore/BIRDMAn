# BIRDMAn

[![GitHub Actions CI](https://github.com/gibsramen/birdman/workflows/BIRDMAn%20CI/badge.svg)](https://github.com/gibsramen/BIRDMAn/actions)
[![Documentation Status](https://readthedocs.org/projects/birdman/badge/?version=latest)](https://birdman.readthedocs.io/en/latest/?badge=latest)

**B**ayesian **I**nferential **R**egression for **D**ifferential **M**icrobiome **An**alysis (**BIRDMAn**) is a framework for performing differential abundance analysis through Bayesian inference.

Much of the code in this repository has been adapted from and inspired by [Jamie Morton](https://mortonjt.github.io/probable-bug-bytes/probable-bug-bytes/differential-abundance/).

## Installation

Currently BIRDMAn requires Python 3.7 or higher.

```bash
conda install -c conda-forge dask biom-format patsy xarray
pip install cmdstanpy
git clone git@github.com:gibsramen/BIRDMAn.git
cd BIRDMAn
pip install .
```

You have to install `cmdstan` through the `cmdstanpy.install_cmdstan` function first.

See the [documentation](https://birdman.readthedocs.io/en/latest/?badge=latest) for details on usage.
