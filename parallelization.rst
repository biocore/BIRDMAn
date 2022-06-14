Parallelization in BIRDMAn through the Single Feature Model
============================================================

There are two ways you can implement a model in BIRDMAn: as a **Table Model** or as a **Single Feature Model**. 

In the **Table Model** method, we initialized one model fitting the entirety of our data. For an example of implementation with the ``TableModel`` class, see our documentation `implementing a custom model <https://github.com/gibsramen/BIRDMAn/blob/main/docs/custom_model.rst>`_.

In the **Single Feature** method, we break down the fitting for each microbe with the ``SingleFeatureModel`` class. Each microbe has its own model fit, the **Single Feature** method allows for computational parallelization. Instead of running the model for each microbe serially as in the **Table Model** method, this process can be optimized to run the model for multiple microbes simultaneously, speeding up uour BIRDMAn workflow.

Check out the Stan manual if you'd like more information on `parallelization in Stan <https://mc-stan.org/docs/2_24/cmdstan-guide/parallelization.html>`_.

To illustrate a walk through of the **Single Feature** method, we will use the same data from our `implementing a custom model
<https://github.com/gibsramen/BIRDMAn/blob/main/docs/custom_model.rst>`_ example. Check out the custom model tutorial for background details on downloading the study data, filtering the metadata, and filtering the feature table. Or, download and save our the filtered ``metadata_model.csv`` and feature table ``filt_tbl.biom``.

Stan code
---------

We will save the below file to ``single_negative_binomial_re.stan`` so we can import and compile it in BIRDMAn. Here, our model code remains similar to the Table Model stan code ``negative_binomial_re.stan`` in the custom model tutorial. The key difference is that we no longer require dimensions with the Single Model approach as we did in the Table model.

.. code-block:: stan

  data {
    int<lower=0> N;                             // number of sample IDs
    int<lower=0> num_subjs;                     // number of groups (subjects)
    int<lower=0> p;                             // number of covariates
    real depth[N];                              // sequencing depths of microbes
    matrix[N, p] x;                             // covariate matrix
    int<lower=0, upper=num_subjs> subj_map[N];  // mapping of samples to subject IDs
    int y[N];                                   // observed microbe abundances
    real<lower=0> B_p;                          // stdev for covariate Beta Normal prior
    real<lower=0> phi_s;                        // scale for dispersion Cauchy prior
    real<lower=0> re_p;                         // stdev for subject intercept Normal prior
  }

  parameters {
    vector[p] beta_var;
    real<lower=0> reciprocal_phi;
    vector[num_subjs] subj_re;
  }

  transformed parameters {
    real phi = 1 / reciprocal_phi;
    vector[N] lam = x*beta_var;

    for (n in 1:N) {
      lam[n] += depth[n] + subj_re[subj_map[n]];
    }
  }

  model {
    beta_var[1] ~ normal(-6, B_p);
    for (i in 2:p) {
      beta_var[i] ~ normal(0, B_p);
    }

    subj_re ~ normal(0, re_p);
    reciprocal_phi ~ cauchy(0, phi_s);

    y ~ neg_binomial_2_log(lam, phi);
  }

  generated quantities {
    vector[N] y_predict;
    vector[N] log_lhood;

    for (n in 1:N) {
      y_predict[n] = neg_binomial_2_log_rng(lam[n], phi);
      log_lhood[n] = neg_binomial_2_log_lpmf(y[n] | lam[n], phi);
    }
  }
      
  
Getting started
----------------
Import the following libraries. If you already have ``birdman`` installed, import the ``SingleFeatureModel`` class. Set your directories to the metadata, biom table, and model code.

.. code-block:: python

  import logging
  import os
  from tempfile import TemporaryDirectory
  import click
  import time
  
  from birdman import SingleFeatureModel
  import biom
  import cmdstanpy
  import numpy as np
  import pandas as pd

  PROJ_DIR = "/path/to/proj/dir"
  MD = pd.read_table(f"{PROJ_DIR}/path/to/metadata_model.csv", sep=",", index_col=0)
  TABLE_FILE = f"{PROJ_DIR}/path/to/filt_tbl.biom" 
  MODEL_PATH = f"{PROJ_DIR}/path/to/single_negative_binomial_re.stan"

  TABLE = biom.load_table(TABLE_FILE)
  FIDS = TABLE.ids(axis="observation")
  
  cmdstanpy.set_cmdstan_path("/path/to/.cmdstan/cmdstan-2.29.1")
  

Define ``SingleFeatureModel`` class
-----------------------------------

.. code-block:: python

  class HelminthModelSingle(SingleFeatureModel):
    def __init__(
        self,
        feature_id: str,
        beta_prior: float = 10.0,
        cauchy_scale: float = 5.0,
        subj_prior: float = 2.0,
        num_iter: int = 500,
        num_warmup: int = 1000,
        **kwargs
    ):
        super().__init__(
            table=TABLE,
            feature_id=feature_id,
            model_path=MODEL_PATH,
            num_iter=num_iter,
            num_warmup=num_warmup,
            **kwargs
        )

        subj_series = MD["host_subject_id"].loc[self.sample_names]
        samp_subj_map = subj_series.astype("category").cat.codes + 1
        self.subjects = np.sort(subj_series.unique())

        formula="C(time_point, Treatment('post-deworm'))" 
        self.create_regression(formula, MD)

        param_dict = {
            "depth": np.log(TABLE.sum(axis="sample")),
            "num_subjs": len(self.subjects),
            "subj_map": samp_subj_map.values,
            "B_p": beta_prior,
            "phi_s": cauchy_scale,
            "re_p": subj_prior
        }
        self.add_parameters(param_dict)

        self.specify_model(
            params=["beta_var", "phi", "subj_re"],
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

  @click.command()
  @click.option("--inference-dir", required=True)
  @click.option("--start-num", required=True)
  @click.option("--end-num", required=True)
  @click.option("--chains", default=4)
  @click.option("--num-iter", default=500)
  @click.option("--num-warmup", default=1000)
  @click.option("--beta-prior", default=10.0)
  @click.option("--cauchy-scale", default=3.0)
  @click.option("--re-prior", default=3.0)
  @click.option("--logfile", required=True)
  
Running BIRDMAn
---------------

.. code-block:: python

  def run_birdman(
      inference_dir,
      start_num,
      end_num,
      chains,
      num_iter,
      num_warmup,
      beta_prior,
      cauchy_scale,
      re_prior,
      logfile,
  ):
      birdman_logger = logging.getLogger("birdman")
      birdman_logger.setLevel(logging.INFO)
      fh = logging.FileHandler(logfile, mode="w")
      sh = logging.StreamHandler()
      formatter = logging.Formatter(
          "[%(asctime)s - %(name)s - %(levelname)s] ::  %(message)s"
      )
      fh.setFormatter(formatter)
      sh.setFormatter(formatter)
      birdman_logger.addHandler(fh)
      birdman_logger.addHandler(sh)

      cmdstanpy_logger = cmdstanpy.utils.get_logger()
      cmdstanpy_logger.addHandler(fh)
      for h in cmdstanpy_logger.handlers:
          h.setFormatter(formatter)

      for feature_num in range(int(start_num), int(end_num)):
          feature_num_str = str(feature_num).zfill(4)
          feature_id = FIDS[feature_num]
          birdman_logger.info(f"Feature num: {feature_num_str}")
          birdman_logger.info(f"Feature ID: {feature_id}")

          tmpdir = f"{inference_dir}/tmp/F{feature_num_str}_{feature_id}"
          outfile = f"{inference_dir}/F{feature_num_str}_{feature_id}.nc"

          os.makedirs(tmpdir, exist_ok=True)

          with TemporaryDirectory(dir=tmpdir) as t:
              model = HelminthModelSingle(
                  feature_id=feature_id,
                  beta_prior=beta_prior,
                  cauchy_scale=cauchy_scale,
                  subj_prior=re_prior,
                  chains=chains,
                  num_iter=num_iter,
                  num_warmup=num_warmup,
              )
              model.compile_model()
              model.fit_model(sampler_args={"output_dir": t})

              inf = model.to_inference_object()
              birdman_logger.info(inf.posterior)

              loo = az.loo(inf, pointwise=True)
              rhat = az.rhat(inf)
              birdman_logger.info("LOO:")
              birdman_logger.info(loo)
              birdman_logger.info("Rhat:")
              birdman_logger.info(rhat)
              if (rhat > 1.05).to_array().any().item():
                  birdman_logger.warning(
                      f"{feature_id} has Rhat values > 1.05"
                  )
              if np.nan in loo.values:
                  birdman_logger.warning(
                      f"{feature_id} has NaN elpd"
                  )

              inf.to_netcdf(outfile)
              birdman_logger.info(f"Saved to {outfile}")
              time.sleep(10)

  if __name__ == "__main__":
      run_birdman()
