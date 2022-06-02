Parallelization in BIRDMAn through the Single Feature model
============================================================

There are two approaches to running a model in BIRDMAn: as a **Table Model** or as a **Single Feature Model**. 

In the **Table Model** method, one model is initialized fitting the entirety of our data. For an example of implementation with the ``TableModel`` class,
see our documentation `implementing a custom model <https://github.com/gibsramen/BIRDMAn/blob/main/docs/custom_model.rst>`_.

In the **Single Feature** method, we break down the fitting for each individual microbe with the ``SingleFeatureModel`` class. 
An advantage of the **Single Feature** method is that it allows for computational parallelization. By paralleling this process, we can speed up the BIRDMAn
modeling fitting process.

Check out the Stan manual for more information on `parallelization in Stan <https://mc-stan.org/docs/2_24/cmdstan-guide/parallelization.html>`_.

To illustrate a walk through of the **Single Feature** method, we will use the same data from our `implementing a custom model
<https://github.com/gibsramen/BIRDMAn/blob/main/docs/custom_model.rst>`_ example. Check out the custom model tutorial for background details on downloading the study data, 
filtering the metadata, and filtering the feature table. Save the filtered metadata and feature table (we will use them in this tutorial).

Stan code
---------

We will save the below file to ``single_negative_binomial_re.stan`` so we can import and compile it in BIRDMAn.

.. code-block:: stan

  data {
    int<lower=1> N;                             // number of sample IDs
    int<lower=1> num_subjs;                     // number of groups (subjects)                  
    int<lower=1> p;                             // number of covariates
    real depth[N];                              // sequencing depths of microbes
    matrix[N, p] x;                             // covariate matrix
    int<lower=1, upper=num_subjs> subj_map[N];  // mapping of samples to subject IDs
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
  
