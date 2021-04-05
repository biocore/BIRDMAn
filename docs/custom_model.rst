Implementing a custom model
===========================

One of the core features of BIRDMAn is that it is extensible to user-defined differential abundance models. This means that BIRDMAn has been designed from the beginning to support custom analysis. Custom modeling is implemented through user-written Stan files.

Here we will walk through a simple custom model and how to incorporate it into BIRDMAn. We will use data from the study "Linking the effects of helminth infection, diet and the gut microbiota with human whole-blood signatures (repeated measurements)" (Qiita ID: 11913). Samples were taken from subjects before and after a deworming procedure.

This paired design introduces a wrinkle in traditional differential abundance analysis: `pseudoreplication <https://en.wikipedia.org/wiki/Pseudoreplication>`_. It is statistically inadmissible to fail to account for repeated measurements as in the case of longitudinal data. One can account for this as a fixed effect in other tools, but really subject-level variation is likely better modeled as a random effect. For this example we will specify a linear mixed-effects (LME) model where ``subject`` is a random intercept.

Downloading data from Qiita
-------------------------------

First we will download the BIOM table and metadata from Qiita.

.. code-block:: bash

    wget -O data.zip "https://qiita.ucsd.edu/public_artifact_download/?artifact_id=44773"
    wget -O metadata.zip "https://qiita.ucsd.edu/public_download/?data=sample_information&study_id=107"
    unzip data.zip
    unzip metadata.zip

We then import the data into Python so we can use BIRDMAn.

.. code-block:: python

    import biom
    import pandas as pd

    table = biom.load_table("BIOM/94270/reference-hit.biom")
    metadata = pd.read_csv(
        "templates/11913_20191016-112545.txt",
        sep="\t",
        index_col=0
    )

Processing metadata
-------------------

We will now determine which subjects in the metadata have samples at both pre and post-deworm timepoints.

.. code-block:: python

    subj_is_paired = (
        metadata
        .groupby("host_subject_id")
        .apply(lambda x: (x["time_point"].values == [1, 2]).all())
    )
    paired_subjs = subj_is_paired[subj_is_paired].index
    paired_samps = metadata[metadata["host_subject_id"].isin(paired_subjs)].index
    cols_to_keep = ["time_point", "host_subject_id"]
    metadata_model = metadata.loc[paired_samps, cols_to_keep].dropna()
    metadata_model["time_point"] = (
        metadata_model["time_point"].map({1: "pre-deworm", 2: "post-deworm"})
    )
    metadata_model["host_subject_id"] = "S" + metadata["host_subject_id"].astype(str)
    metadata_model.head()

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - sample-name
      - time_point
      - host_subject_id
    * - 11913.102
      - pre-deworm
      - S102
    * - 11913.102AF
      - post-deworm
      - S102
    * - 11913.1097
      - pre-deworm
      - S1097
    * - 11913.1097AF
      - post-deworm
      - S1097
    * - 11913.119
      - pre-deworm
      - S119

Filtering feature table
-----------------------

This table has nearly 3000 features, most of which are likely lowly prevalent. We will first filter the table to only include the samples we previously described and then filter out features that are present in fewer than 5 samples.

.. code-block:: python

    raw_tbl_df = table.to_dataframe()
    samps_to_keep = sorted(list(set(raw_tbl_df.columns).intersection(metadata_model.index)))
    filt_tbl_df = raw_tbl_df.loc[:, samps_to_keep]
    prev = filt_tbl_df.clip(upper=1).sum(axis=1)
    filt_tbl_df = filt_tbl_df.loc[prev[prev >= 5].index, :]
    filt_tbl = biom.table.Table(
        filt_tbl_df.values,
        sample_ids=filt_tbl_df.columns,
        observation_ids=filt_tbl_df.index
    )

We now have a table of 269 features by 46 samples (23 subjects). This is a much more manageable size!

Model specification
-------------------

For this custom model we want to specify that ``time_point`` is a fixed effect and ``host_subject_id`` is a random effect. We are keeping this model relatively simple but you can imagine a more complicated model with random slopes, specified covariance structures, etc. Our model can thus be written as follows:

.. math::

    y_{ij} &\sim \textrm{NB}(\mu_{ij},\phi_j)

    \mu_{ij} &= n_i p_{ij}

    \textrm{alr}^{-1}(p_i) &= x_i \beta + z_i u

Where :math:`z_i` represents the mapping of sample :math:`i` to subject and :math:`u` represents the subject coefficient vector.

We also specify the following priors:

.. math::

    \beta_j &\sim \textrm{Normal}(0, B_p), B_p \in \mathbb{R}_{>0}

    \frac{1}{\phi_j} &\sim \textrm{Cauchy}(0, C_s), C_s \in
        \mathbb{R}_{>0}

    u_i &\sim \textrm{Normal}(0, u_p), u_p \in \mathbb{R}_{>0}


Stan code
---------

We will save the below file to ``negative_binomial_re.stan`` so we can import and compile it in BIRDMAn.


.. code-block:: stan

    data {
        int<lower=0> N;                     // number of sample IDs
        int<lower=0> S;                     // number of groups (subjects)
        int<lower=0> D;                     // number of dimensions
        int<lower=0> p;                     // number of covariates
        real depth[N];                      // sequencing depths of microbes
        matrix[N, p] x;                     // covariate matrix
        int<lower=1, upper=S> subj_ids[N];  // mapping of samples to subject IDs
        int y[N, D];                        // observed microbe abundances
        real<lower=0> B_p;                  // stdev for covariate Beta Normal prior
        real<lower=0> phi_s;                // scale for dispersion Cauchy prior
        real<lower=0> u_p;                  // stdev for subject intercept Normal prior
    }

    parameters {
        matrix[p, D-1] beta;
        vector<lower=0>[D] reciprocal_phi;
        vector[S] subj_int;
    }

    transformed parameters {
        matrix[N, D-1] lam;
        matrix[N, D] lam_clr;
        vector<lower=0>[D] phi;

        for (i in 1:D){
            phi[i] = 1. / reciprocal_phi[i];
        }

        lam = x*beta;  // N x D-1
        for (n in 1:N){
            lam[n] += subj_int[subj_ids[n]];
        }
        lam_clr = append_col(to_vector(rep_array(0, N)), lam);
    }

    model {
        // setting priors ...
        for (i in 1:D){
            reciprocal_phi[i] ~ cauchy(0., phi_s);
        }
        for (i in 1:D-1){
            for (j in 1:p){
                beta[j, i] ~ normal(0., B_p); // uninformed prior
            }
        }
        for (i in 1:S){
            subj_int[i] ~ normal(0., u_p);
        }

        // generating counts
        for (n in 1:N){
            for (i in 1:D){
                target += neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi[i]);
            }
        }
    }

    generated quantities {
        matrix[N, D] y_predict;
        matrix[N, D] log_lik;

        for (n in 1:N){
            for (i in 1:D){
                y_predict[n, i] = neg_binomial_2_log_rng(depth[n] + lam_clr[n, i], phi[i]);
                log_lik[n, i] = neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi[i]);
            }
        }
    }

Running BIRDMAn
---------------

We will now pass this file along with our table, metadata, and formula into BIRDMAn. Note that we are using the base ``Model`` class for our custom model.

.. code-block:: python

    import birdman

    nb_lme = birdman.Model(  # note that we are instantiating a base Model object
        table=filt_tbl,
        formula="C(time_point, Treatment('pre-deworm'))",
        metadata=metadata_model.loc[samps_to_keep],
        model_path="negative_binomial_re.stan",
        num_iter=500,
        chains=4,
        seed=42
    )

We then want to update our data dictionary with the new parameters.

By default BIRDMAn computes and includes:

* ``y``: table data
* ``x``: covariate design matrix
* ``N``: number of samples
* ``D``: number of features
* ``p``: number of covariates (including Intercept)

We want to add the necessary variables to be passed to Stan:

* ``S``: total number of groups (subjects)
* ``subj_ids``: mapping of samples to subject
* ``B_p``: stdev prior for normally distributed covariate-feature coefficients
* ``phi_s``: scale prior for half-Cauchy distributed overdispersion coefficients
* ``depth``: log sampling depths of samples
* ``u_p``: stdev prior for normally distributed subject intercept shifts

We want to provide ``subj_ids`` with a mapping of which sample corresponds to which subject. Stan does not understand strings so we encode each unique subject as an integer (starting at 1 because Stan 1-indexes arrays).

.. code-block:: python

    import numpy as np

    group_var_series = metadata_model.loc[samps_to_keep]["host_subject_id"]
    samp_subj_map = group_var_series.astype("category").cat.codes + 1
    groups = np.sort(group_var_series.unique())

Now we can add all the necessary parameters to BIRDMAn with the ``add_parameters`` function.

.. code-block:: python

    param_dict = {
        "S": len(groups),
        "subj_ids": samp_subj_map.values,
        "depth": np.log(filt_tbl.sum(axis="sample")),
        "B_p": 3.0,
        "phi_s": 3.0,
        "u_p": 1.0
    }
    nb_lme.add_parameters(param_dict)

Finally, we compile and fit the model.

.. note::

    Fitting this model took approximately 6 minutes on my laptop.

.. code-block:: python

    nb_lme.compile_model()
    nb_lme.fit_model()

Converting to ``InferenceData``
-------------------------------

With a custom model there is a bit more legwork involved in converting to the ``arviz.InferenceData`` data structure. We will step through each of the parameters in this example.

* ``params``: List of parameters you want to include in the posterior draws (must match Stan code).
* ``dims``: Dictionary of dimensions of each parameter to include. Note that we also include the names of the variables for log likelihood and posterior predictive values, ``log_lik`` and ``y_predict`` respectively.
* ``coords``: Mapping of dimensions in ``dims`` to their indices. We internally save ``feature_names``, ``sample_names``, and ``colnames`` (names of covariates in design matrix).
* ``alr_params``: List of parameters which come out of Stan in ALR coordinates. These will be converted into centered CLR coordinates before being returned.
* ``posterior_predictive``: Name of variable holding posterior predictive values (optional).
* ``log_likelihood``: Name of variable holding log likelihood values (optional).
* ``include_observed_data``: Whether to include the original feature table as a group. This is useful for certain diagnostics.

.. code-block:: python

    inference = nb_lme.to_inference_object(
        params=["beta", "phi", "subj_int"],
        dims={
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
            "subj_int": ["subject"],
            "log_lik": ["tbl_sample", "feature"],
            "y_predict": ["tbl_sample", "feature"]
        },
        coords={
            "feature": nb_lme.feature_names,
            "covariate": nb_lme.colnames,
            "subject": groups,
            "tbl_sample": nb_lme.sample_names
        },
        alr_params=["beta"],
        posterior_predictive="y_predict",
        log_likelihood="log_lik",
        include_observed_data=True
    )

With this you can use the rest of the BIRDMAn suite as usual or directly interact with the ``arviz`` library!
