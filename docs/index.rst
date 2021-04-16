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

.. note:: BIRDMAn requires Python >= 3.7

There are several dependencies you must install to use BIRDMAn.
The easiest way to install the required dependencies is through ``conda``.

.. code:: bash

    conda install -c conda-forge dask biom-format patsy xarray
    pip install cmdstanpy
    pip install git+git://github.com/arviz-devs/arviz.git
    pip install birdman

If you are planning on contributing to BIRDMAn you must also install the following packages:

* `pytest <https://docs.pytest.org/en/stable/>`_
* `scikit-bio <http://scikit-bio.org/>`_
* `sphinx <https://www.sphinx-doc.org/en/master/>`_

**Installing cmdstan**

When you install BIRDMAn, you must also install the C++ toolchain that allows ``cmdstanpy`` to run ``cmdstan``.

.. note:: At the time of writing there is a bug in version 2.26.0 of ``cmdstan`` so we install the previous version 2.25.0 instead.

In Python:

.. code:: python

    import cmdstanpy
    cmdstanpy.install_cmdstan(version='2.25.0')

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    default_model_example
    custom_model
    parallelization
    diagnosing_model
    writing_stan_code
    working_with_arviz

.. toctree::
    :maxdepth: 2
    :caption: How BIRDMAn Works

    bayesian_inference

.. toctree::
    :maxdepth: 2
    :caption: API

    models
    diagnostics
    stats
    util
    visualization

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
