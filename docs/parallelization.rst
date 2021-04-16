Fitting multiple features in parallel
=====================================

The default behavior in BIRDMAn is to fit 4 Markov chains in parallel for model fitting. With the high-dimensional nature of 'omics data, this can end up taking a long time. We provide an alternate way to model fitting in which you parallelize across *features* rather than chains. This allows for much faster computation if you have access to multiple cores such as in a HPC environment.

.. note::

    At the moment feature paralellization is only supported in the NegativeBinomial model by default. If you are implementing a custom model that works with parallelization you can still fit in this manner. Any model that can be decomposed into fitting individual models for each feature should work.

We use `dask <https://docs.dask.org/en/latest/>`_ and `dask-jobqueue <https://jobqueue.dask.org/en/latest/>`_ to perform parallelization as they integrate with HPC very nicely. Please see the documentation of these two packages for more details.

In this example we will assume that we have access to a SLURM cluster on which we can run BIRDMAn.

.. code-block:: python

    import biom
    from birdman import NegativeBinomial
    from dask_jobqueue import SLURMCluster
    import pandas as pd

    table = biom.load_table("example.biom")
    metadata = pd.read_csv("metadata.tsv", sep="\t", index_col=0)

    cluster = SLURMCluster(
        cores=2,
        processes=2,
        memory='10GB',
        walltime='01:00:00',
        local_directory='/scratch'
    )

    nb = NegativeBinomial(
        table=table,
        metadata=metadata,
        formula="diet",
        chains=4,
        seed=42,
        parallelize_across="features"  # note this argument!
    )
    nb.compile_model()
    nb.fit_model(dask_cluster=cluster, jobs=8)  # run 8 microbes at a time

This code will run 8 microbes in parallel which will be faster than running the 4 chains in parallel. You can scale this up further depending on your computational resources by increasing the ``jobs`` argument.

When fitting in parallel, CmdStanPy fits a model for each feature in your table. Converting to an ``InferenceData`` object is then just a matter of combining all the individual fits. We can also parallelize this process to speed-up runtime.

.. code-block:: python

    nb.to_inference_object(dask_cluster=cluster, jobs=8)
