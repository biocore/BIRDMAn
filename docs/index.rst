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
The easiest way to install the required dependencies is through ``conda`` or ``mamba``.

.. code:: bash

    conda install -c conda-forge biom-format patsy xarray arviz cmdstanpy
    pip install birdman

If you are planning on contributing to BIRDMAn you must also install the following packages:

* `pytest <https://docs.pytest.org/en/stable/>`_
* `scikit-bio <http://scikit-bio.org/>`_
* `sphinx <https://www.sphinx-doc.org/en/master/>`_

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    default_model_example
    custom_model
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
    summary
    transform
    stats
    visualization

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
