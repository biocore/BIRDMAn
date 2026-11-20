Installing cmdstanpy
====================

BIRDMAn uses the popular ``cmdstanpy`` interface for Bayesian inference. ``cmdstanpy`` is simply a lightweight Python wrapper around the ``cmdstan`` C++ toolchain.

Depending on how you install ``cmdstanpy``, there may be some additional steps you need to take. We recommend installing via conda/mamba if possible because this will automatically install ``cmdstan`` to your system.

.. code-block:: bash

   mamba install -c conda-forge cmdstanpy

If you are installing via pip, you need to install ``cmdstan`` manually. This is because pip cannot install the C++ components on its own.

First, install ``cmdstanpy`` as follows:

.. code-block:: bash

   pip install cmdstanpy

Then, from your terminal you can run the ``install_cmdstan`` script to install the C++ toolchain.

.. code-block:: bash

   install_cmdstan

For more information on installing ``cmdstanpy``, please see the `documentation <https://mc-stan.org/cmdstanpy/installation.html>`_.

Runtime and resources
---------------------

BIRDMAn runs on a standard laptop. With the single-feature models each feature is fit independently, using well under 1 GB of memory. On one CPU core, a negative binomial fit with 4 MCMC chains of 500 warmup and 500 sampling iterations takes about a second for a few hundred samples; a model with several subject-level random effects takes 10–20 seconds. Total runtime grows with the number of features, and features can be fit in parallel (see :doc:`parallelization`).
