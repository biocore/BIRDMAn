Writing Stan code
=================

The core of BIRDMAn is running compiled Stan code that defines your model. While we do provide several default options, one of the key strengths of BIRDMAn is its extensibility to custom differential abundance modeling. This page gives an overview of how to write custom Stan code for BIRDMAn modeling.

See the `Stan documentation <https://mc-stan.org/docs/2_26/stan-users-guide/index.html>`_ for more details.

Structure of a Stan program
---------------------------

Stan programs are made up of several "blocks". The main ones we use in BIRDMAn are:

* ``data``: This block defines the types and dimensions of the data used in the program. Every variable present in this block must be passed into the Stan program from Python.
* ``parameters``: This block defines the types and dimensions of the parameters you are interested in fitting.
* ``transformed parameters``: This block defines any transformations to perform on the data. For example we use this block to multiply the design matrix by the feature-covariate coefficient matrix.
* ``model``: This block defines the setting of any priors and the fitting of your target distribution (e.g. negative binomial).
* ``generated quantities``: This block is for generating any other results such as posterior predictive values and log likelihood values.

Data
----

Every Stan program you use with BIRDMAn passes the following data by default into the data block:

* ``N``: Number of samples
* ``D``: Number of features
* ``x``: Design matrix
* ``p``: Number of columns in the design matrix (computed from formula)
* ``y``: Feature table values

Any other parameters must be added to the BIRDMAn model with the ``add_parameters`` method (unless using a default model).

Parameters
----------

In this block you should specify the parameters you are primarily interested in fitting. This includes coefficient matrices, random intercepts, etc. Important to note that if you are fitting feature coefficients you may need to use ALR coordinates. For example, in the ``NegativeBinomial`` default model we specify ``matrix[p, D-1] beta;`` as ALR coordinates remove one feature as reference.

Transformed parameters
----------------------

The ``transformed parameters`` block is used for defining operations on the parameters more advanced than just defining them. In the default models we use it to wrangle :math:`x\beta` into a format that's easier to use in the ``model`` block.

Model
-----

This is where the meat of the computation lies. The model block is where you assign priors and compute likelihood. The parameters and data should all come together in this block to sample from the posterior for your regression.

Generated quantities
--------------------

This block is primarily for calculations in addition to HMC sampling. In BIRDMAn default models we perform posterior predictive and log likelihood calculations in this block from the draws generated in the model block.
