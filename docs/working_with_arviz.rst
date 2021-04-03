Working with ``arviz``
======================

``arviz`` is a wonderful library for analysis of Bayesian models. It includes many functions to help researchers determine how well a model performs both visually and numerically. However, those coming from primarily ``pandas`` and ``numpy`` experience may find it a bit challenging to orient themselves to using ``arviz`` and its underlying data structure derived from the ``xarray`` package. Here, we give a short breakdown on how to start analyzing an ``arviz.InferenceData`` object.

For more information on these two packages, see their documentation:

* `arviz <https://arviz-devs.github.io/arviz/index.html>`_
* `xarray <http://xarray.pydata.org/en/stable/>`_

``xarray``
----------

An ``arviz.InferenceData`` is essentially a collection of ``xarray`` objects so we will cover ``xarray`` first. ``xarray`` is a Python package designed for efficient and elegant interaction of multi-dimensional data. In Bayesian analysis we deal primarily with data that includes both chain and draw as dimensions in addition to the original parameter dimensions. For example, in a ``birdman.NegativeBinomial`` model, the :math:`\beta` parameter will be output with 4 dimensions: chain, draw, covariate, and feature. Dealing with all of these dimensions in simple NumPy arrays can get confusing as you try to keep track of which dimension is which. ``xarray`` uses named dimensions and coordinates to make this process much cleaner and intuitive.

``xarray.Dataset``
^^^^^^^^^^^^^^^^^^

The ``Dataset`` is the primary data structure that you will be interfacing with in ``xarray``. A ``Dataset`` is a collection of data variables (for example parameters from posterior sampling) that can have different dimensionality. As an example, here is a ``Dataset`` holding posterior draws from a ``birdman.NegativeBinomial`` model:

.. code-block::

    <xarray.Dataset>
    Dimensions:    (chain: 4, covariate: 2, draw: 100, feature: 143)
    Coordinates:
    * chain      (chain) int64 0 1 2 3
    * draw       (draw) int64 0 1 2 3 4 5 6 7 8 9 ... 91 92 93 94 95 96 97 98 99
    * feature    (feature) object '193358' '4465746' ... '212228' '192368'
    * covariate  (covariate) object 'Intercept' 'diet[T.DIO]'
    Data variables:
        beta       (chain, draw, covariate, feature) float64 7.142 3.549 ... 0.4587
        phi        (chain, draw, feature) float64 0.1216 0.3286 ... 0.8205 0.6151
    Attributes:
        created_at:                 2021-04-01T22:37:03.663497
        arviz_version:              0.11.2
        inference_library:          cmdstanpy
        inference_library_version:  0.9.68

The ``Dimensions`` descriptor shows the names and number of entries in each dimensions. The ``Coordinates`` entry holds the labels for each of the dimensions. In this example we see that the chains are labeled 0-3 and the draws are labeled 0-99. However, the features are labeled with OTU IDs and the covariates are labeled with the entries in the design matrix. The ``Data variables`` entry contains the actual data (in this case parameters) and lists the dimensionality of each. Note that ``beta`` is of dimension chain, draw, covariate, feature while ``phi`` is only of dimension chain, draw, feature. The ``Dataset`` is a powerful data structure because it can hold data variables of varying dimensionality.

``xarray.DataArray``
^^^^^^^^^^^^^^^^^^^^

A ``Dataset`` is simply a collection of ``DataArray`` objects. Whereas a ``Dataset`` can contain multiple data variables, a ``DataArray`` contains only one. If you want to access the ``beta`` variable from the above ``Dataset``, you simply index it like you would a dictionary. If you have a ``Dataset``, ``ds``, with a data variable ``beta``, you would access it with ``ds["beta"]`` which returns:

.. code-block::

    <xarray.DataArray 'beta' (chain: 4, draw: 100, covariate: 2, feature: 143)>
    array([[[[ 7.14216 , ..., -0.03673 ],
            [-0.639199, ...,  0.958811]],

            ...,

            [[ 7.071049, ..., -0.374691],
            [-0.61982 , ...,  0.5009  ]]],


            ...,


            [[[ 7.096262, ..., -0.281968],
            [-0.607823, ...,  1.190807]],

            ...,

            [[ 7.185024, ...,  0.038614],
            [-0.671318, ...,  0.458722]]]])
    Coordinates:
    * chain      (chain) int64 0 1 2 3
    * draw       (draw) int64 0 1 2 3 4 5 6 7 8 9 ... 91 92 93 94 95 96 97 98 99
    * feature    (feature) object '193358' '4465746' ... '212228' '192368'
    * covariate  (covariate) object 'Intercept' 'diet[T.DIO]'

Selecting and indexing data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manipulating data in ``xarray`` is a bit more involved than in NumPy. The most important thing to keep in mind is the notion of ``dims`` (dimensions) and ``coords`` (coordinates). Dimensions are the *names* while coordinates are the *labels*.

You can use the ``.sel`` function to select specific slices of data. To extract all values from just chain 0, you would run:

.. code-block:: python

    ds["beta"].sel(chain=0)

You can also index across multiple dimensions - if you wanted only values from chain 2 from the diet covariate you would run:

.. code-block:: python

    ds["beta"].sel(chain=2, covariate="diet[T.DIO]")

This also works with multiple values for a given dimension. As an example if you wanted to get all diet posterior samples from just features 193358 and 4465746 you would run:

.. code-block:: python

    ds["beta"].sel(feature=["193358", "4465746"], covariate="diet[T.DIO]")

See the `documentation <http://xarray.pydata.org/en/stable/indexing.html>`_ for more on indexing and selecting data.

``arviz``
---------

An ``arviz.InferenceData`` object is a collection of ``xarray.Datasets`` organized for use in Bayesian model analysis. Each inference comprises several groups such as posterior draws, sample stats, log likelihood values, etc. ``arviz`` organizes these different groups such that they can be used seamlessly for downstream analysis.

If you run a ``birdman.NegativeBinomial`` model and convert it to an inference object, you can print this object and see the following:

.. code-block:: bash

    Inference data with groups:
            > posterior
            > posterior_predictive
            > log_likelihood
            > sample_stats
            > observed_data

Each group is an ``xarray.Dataset`` that you can interact with as described above. You can access each of these groups with either attribute notation (``inference.posterior``) or index notation (``inference["posterior"]``).

Saving and loading data
^^^^^^^^^^^^^^^^^^^^^^^

It is useful to be able to save the results of BIRDMAn so that they can be analyzed later or distributed to collaborators. The best way to do this is to save the ``InferenceData`` object in the `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ format. This is a compressed format that works very well with multi-dimensional arrays.

You can save and load fitted models with ``to_netcdf`` and ``from_netcdf``.

.. code-block:: python

    import arviz as az
    inference.to_netcdf("inference.nc")
    inference_loaded = az.from_netcdf("inference.nc")
