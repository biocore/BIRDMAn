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
