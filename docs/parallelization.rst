Parallelization in BIRDMAn through the Single Feature Model
============================================================

There are two ways you can implement a model in BIRDMAn: as a **Table Model** or as a **Single Feature Model**. 

In the **Table Model** method, we initialized one model fitting the entirety of our data. For an example of implementation with the ``TableModel`` class, see our documentation `implementing a custom model <https://github.com/gibsramen/BIRDMAn/blob/main/docs/custom_model.rst>`_.

In the **Single Feature** method, we break down the fitting for each microbe with the ``SingleFeatureModel`` class. Each microbe has its own model fit, the **Single Feature** approach allows for computational parallelization. Instead of running the model for each microbe serially as in the **Table Model** method, this process can be optimized to run the model for multiple microbes simultaneously, speeding up your BIRDMAn workflow.

To illustrate a walk through of the **Single Feature** method, we will use the same data from our `implementing a custom model
<https://github.com/gibsramen/BIRDMAn/blob/main/docs/custom_model.rst>`_ example. Check out the custom model tutorial for background details on downloading the study data, filtering the metadata, and filtering the feature table. Or, download and save our the filtered ``metadata_model.csv`` and feature table ``filt_tbl.biom``.

Stan code
---------

We will save the below file to ``single_negative_binomial_re.stan`` so we can import and compile it in BIRDMAn. Here, our model code remains similar to the Table Model stan code ``negative_binomial_re.stan`` in the custom model tutorial. The key difference is that we no longer require dimensions with the Single Model approach as we did in the Table model.

.. code-block:: stan

    data {
      int<lower=1> N;                                   // number of sample IDs
      int<lower=1> num_subjs;                           // number of groups (subjects)
      int<lower=1> p;                                   // number of covariates
      real A;                                           // mean of intercept prior
      vector[N] log_depths;                             // sequencing depths of microbes
      matrix[N, p] x;                                   // covariate matrix
      array[N] int<lower=0, upper=num_subjs> subj_map;  // mapping of samples to subject IDs
      array[N] int y;                                   // observed microbe abundances
      real<lower=0> B_p;                                // stdev for covariate beta normal prior
      real<lower=0> inv_disp_sd;                        // stdev for inverse dispersion lognormal prior
      real<lower=0> re_p;                               // stdev for subject intercept normal prior
    }

    parameters {
      real<offset=A, multiplier=2> beta_0;              // intercept parameter
      vector[p-1] beta_x;                               // parameters for covariates
      real<lower=0> inv_disp;                           // inverse dispersion parameter
      vector[num_subjs] subj_re;                        // subject intercepts
    }

    transformed parameters {
      vector[p] beta_var = append_row(beta_0, beta_x);
      vector[N] lam = x*beta_var + log_depths;

      for (n in 1:N) {
        // add subject intercepts
        lam[n] += subj_re[subj_map[n]];
      }
    }

    model {
      // Specify priors
      beta_0 ~ normal(A, 2);
      for (i in 1:p-1) {
        beta_x[i] ~ normal(0, B_p);
      }

      subj_re ~ normal(0, re_p);
      inv_disp ~ lognormal(0, inv_disp_sd);

      // Fit model
      y ~ neg_binomial_2_log(lam, inv(inv_disp));
    }

    generated quantities {
      vector[N] y_predict;  // posterior predictive model
      vector[N] log_lhood;  // Evaluate log-likelihood of samples from posterior

      for (n in 1:N) {
        y_predict[n] = neg_binomial_2_log_rng(lam[n], inv(inv_disp));
        log_lhood[n] = neg_binomial_2_log_lpmf(y[n] | lam[n], inv(inv_disp));
      }
    }


Getting started
----------------
Import the following libraries. If you already have ``birdman`` installed, import the ``SingleFeatureModel`` class. Set your directories to the metadata, biom table, and model code.

.. code-block:: python

  import os
  from tempfile import TemporaryDirectory
  import click

  from birdman import SingleFeatureModel, ModelIterator
  import biom
  import numpy as np
  import pandas as pd

  PROJ_DIR = "/path/to/proj/dir"
  MD = pd.read_table(f"{PROJ_DIR}/path/to/metadata_model.csv", sep=",", index_col=0)
  TABLE_FILE = f"{PROJ_DIR}/path/to/filt_tbl.biom"
  MODEL_PATH = f"{PROJ_DIR}/path/to/single_negative_binomial_re.stan"

  TABLE = biom.load_table(TABLE_FILE)
  FIDS = TABLE.ids(axis="observation")

Define ``SingleFeatureModel`` class
-----------------------------------
We will now pass this file along with our table, metadata, and formula into BIRDMAn. Note that we are using the base ``SingleFeatureModel`` class for our model. We inherit this class to build our own custom model.

.. code-block:: python

  class HelminthModelSingle(SingleFeatureModel):
    def __init__(
        self,
        table: biom.Table,
        feature_id: str,
        beta_prior: float = 2.0,
        inv_disp_sd: float = 0.5,
        subj_prior: float = 2.0,
        num_iter: int = 500,
        num_warmup: int = 1000,
    ):
        super().__init__(
            table=table,
            feature_id=feature_id,
            model_path=MODEL_PATH,
            num_iter=num_iter,
            num_warmup=num_warmup,
        )

        # Create a mapping of sample to subject
        # Start at 1 because Stan 1-indexes
        subj_series = MD["host_subject_id"].loc[self.sample_names]
        samp_subj_map = subj_series.astype("category").cat.codes + 1
        self.subjects = np.sort(subj_series.unique())

        # Create the design matrix
        formula="C(time_point, Treatment('post-deworm'))"
        self.create_regression(formula, MD)

        # Assume intercept prior is log-average proportion based on total number of features
        D = table.shape[0]
        A = np.log(1 / D)

        param_dict = {
            "log_depths": np.log(TABLE.sum(axis="sample")),
            "A": A,
            "num_subjs": len(self.subjects),
            "subj_map": samp_subj_map.values,
            "B_p": beta_prior,
            "inv_disp_sd": inv_disp_sd,
            "re_p": subj_prior
        }
        self.add_parameters(param_dict)

        # Specify the parameters you want to keep and dimensions
        self.specify_model(
            params=["beta_var", "inv_disp", "subj_re"],
            dims={
                "beta_var": ["covariate"],
                "subj_re": ["subject"],
                "log_lhood": ["tbl_sample"],
                "y_predict": ["tbl_sample"]
            },
            coords={
                "covariate": self.colnames,
                "tbl_sample": self.sample_names,
                "subject": self.subjects
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )

Chunking
--------

Now that we have created our model class, we want to write a script to assist us with parallelization. BIRDMAn provides a convenience class, ``ModelIterator``, to simplify fitting multiple features at once. This class allows us to "chunk" our feature table so that we can run subsets of the table on different cores.

When you pass in the number of chunks, BIRDMAn automatically determines how many features per chunk and creates a list of lists of tuples. For example, if we have a table of 8 features that we want to split into three chunks, ``ModelIterator`` would subset the table as follows:

.. code-block:: python

    [
        [(feature_1, model_1), (feature_2, model_2), (feature_3, model_3)],
        [(feature_4, model_4), (feature_5, model_5), (feature_6, model_6)],
        [(feature_7, model_7), (feature_8, model_8)]
    ]

What this means for us is that we can write a script that takes in the total number of chunks and current chunk number and fits all the features in that chunk. With a HPC such as SLURM or Torque, we can use multiple cores to significantly speed up the time to fit all features! We will call this Python script ``run_birdman_chunked.py``. We will use the click package to create a simple CLI.

.. code-block:: python

    @click.command()
    @click.option("--inference-dir", required=True)
    @click.option("--num-chunks", required=True)
    @click.option("--chunk-num", required=True)
    @click.option("--chains", default=4)
    @click.option("--num-iter", default=500)
    @click.option("--num-warmup", default=1000)
    @click.option("--beta-prior", default=2.0)
    @click.option("--inv-disp-sd", default=0.5)
    @click.option("--re-prior", default=2.0)
    def run_birdman(
        inference_dir,
        num_chunks,
        chunk_num,
        chains,
        num_iter,
        num_warmup,
        beta_prior,
        inv_disp_sd,
        re_prior,
    ):
        model_iter = ModelIterator(
            TABLE,
            HelminthModelSingle,
            num_chunks=num_chunks,
            beta_prior=beta_prior,
            inv_disp_sd=inv_disp_sd,
            subj_prior=re_prior,
            chains=chains,
            num_iter=num_iter,
            num_warmup=num_warmup
        )
        # Get chunk number - array job starts at 1 so we subtract for indexing
        chunk = model_iter[chunk_num - 1]

        for feature_id, model in chunk:
            # Specify a temporary directory for temporary files during model fitting
            tmpdir = f"{inference_dir}/tmp/{feature_id}"
            os.makedirs(tmpdir, exist_ok=True)

            # Specify the output file
            outfile = f"{inference_dir}/{feature_id}.nc"

            with TemporaryDirectory(dir=tmpdir) as t:
                model.compile_model()
                model.fit_model(sampler_args={"output_dir": t})

                inf = model.to_inference()
                inf.to_netcdf(outfile)

    if __name__ == "__main__":
        run_birdman()

Running our parallel script
---------------------------

With our Python script, we can now write a simple script that tells our cluster how to chunk our workflow and allocate resources. For this tutorial we will be using SLURM but there should be equivalent procedures in other schedulers. The ``--array=1-20`` line indicates that we want to create 20 chunks of our table.

.. note::

    You should compile your custom Stan file before you run this script. Otherwise, each instance will try to compiile the model and run into issues. You can do this easily with ``cmdstanpy.CmdStanModel(stan_file="single_negative_binomial_re.stan")``.

.. code-block:: bash

    #!/bin/bash
    #SBATCH --mem=8G
    #SBATCH --nodes=1
    #SBATCH --partition=short
    #SBATCH --cpus-per-task=4
    #SBATCH --time=6:00:00
    #SBATCH --array=1-20

    echo Chunk $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX

    OUTDIR="./inferences"
    mkdir $OUTDIR

    echo Starting Python script...
    python run_birdman_chunked.py \
        --inference-dir $OUTDIR \
        --num-chunks $SLURM_ARRAY_TASK_MAX \
        --chunk-num $SLURM_ARRAY_TASK_ID \
        --chains 4 \
        --num-iter 500 \
        --num-warmup 1000 \
        --beta-prior 2.0 \
        --inv-disp-sd 0.5 \
        --re-prior 2.0 \

If you run this script your scheduler should start up 20 jobs, each one running a different chunk of features!
