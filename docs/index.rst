.. BIRDMAn documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:24:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

   Much of this documentation was inspired by the UMAP docs.

BIRDMAn: Bayesian Inferential Regression for Differential Microbiome Analysis
=============================================================================

BIRDMAn is a framework for performing differential abundance on microbiome data through a Bayesian lens.
We provide several default models but also allow users to create their own statistical models based on their specific experimental designs/questions.

Installation
------------

There are several dependencies you must install to use BIRDMAn.
The easiest way to install the required dependencies is through ``conda``.

.. code:: bash

    conda install -c conda-forge dask biom-format patsy xarray
    git clone git@github.com:gibsramen/BIRDMAn.git
    cd BIRDMAn
    pip install -e .

If you are planning on contributing to BIRDMAn you must also install ``pytest`` and ``scikit-bio``.

**Installing cmdstan**

When you install BIRDMAn, you must also install the C++ toolchain that allows ``cmdstanpy`` to run ``cmdstan``.

.. note:: At the time of writing there is a bug in version 2.26.0 of ``cmdstan`` so we install the previous version 2.25.0 instead.

In Python:

.. code:: python

    import cmdstanpy
    cmdstanpy.install_cmdstan(version='2.25.0')

.. toctree::
    :maxdepth: 2
    :caption: Tutorial

    default_model_example
    user_defined_model
