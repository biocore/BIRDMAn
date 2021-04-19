import os
from pkg_resources import resource_filename

import arviz as az
import biom
import dask_jobqueue
import numpy as np
import pandas as pd

from .model_base import Model

TEMPLATES = resource_filename("birdman", "templates")
DEFAULT_MODEL_DICT = {
    "negative_binomial": {
        "chains": os.path.join(TEMPLATES, "negative_binomial.stan"),
        "features": os.path.join(TEMPLATES, "negative_binomial_single.stan"),
        "lme": os.path.join(TEMPLATES, "negative_binomial_lme.stan")
    },
    "multinomial": os.path.join(TEMPLATES, "multinomial.stan")
}


class NegativeBinomial(Model):
    """Fit count data using negative binomial model.

    .. math::

        y_{ij} &\\sim \\textrm{NB}(\\mu_{ij},\\phi_j)

        \\mu_{ij} &= n_i p_{ij}

        \\textrm{alr}^{-1}(p_i) &= x_i \\beta

    Priors:

    .. math::

        \\beta_j &\\sim \\textrm{Normal}(0, B_p), B_p \\in \\mathbb{R}_{>0}

        \\frac{1}{\\phi_j} &\\sim \\textrm{Cauchy}(0, C_s), C_s \\in
            \\mathbb{R}_{>0}


    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame

    :param num_iter: Number of posterior sample draws, defaults to 500
    :type num_iter: int

    :param num_warmup: Number of posterior draws used for warmup, defaults to
        num_iter
    :type num_warmup: int

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float

    :param beta_prior: Standard deviation for normally distributed prior values
        of beta, defaults to 5.0
    :type beta_prior: float

    :param cauchy_scale: Scale parameter for half-Cauchy distributed prior
        values of phi, defaults to 5.0
    :type cauchy_scale: float

    :param parallelize_across: Whether to parallelize across features or
        chains, defaults to 'chains'
    :type parallelize_across: str
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
        parallelize_across: str = "chains",
    ):
        filepath = DEFAULT_MODEL_DICT["negative_binomial"][parallelize_across]
        super().__init__(
            table=table,
            formula=formula,
            metadata=metadata,
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
            parallelize_across=parallelize_across
        )

        param_dict = {
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)

    def to_inference_object(
        self,
        dask_cluster: dask_jobqueue.JobQueueCluster = None,
        jobs: int = 4
    ) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :param dask_cluster: Dask jobqueue to run parallel jobs (optional)
        :type dask_cluster: dask_jobqueue.JobQueueCluster, optional

        :param jobs: Number of jobs to run in parallel, defaults to 4
        :type jobs: int

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        dims = {
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
            "log_lhood": ["tbl_sample", "feature"],
            "y_predict": ["tbl_sample", "feature"]
        }
        coords = {
            "covariate": self.colnames,
            "feature": self.feature_names,
            "tbl_sample": self.sample_names
        }

        # TODO: May want to allow not passing PP/LL/OD in the future
        args = dict()
        if self.parallelize_across == "chains":
            args["alr_params"] = ["beta"]
        else:
            args["dask_cluster"] = dask_cluster
            args["jobs"] = jobs
        inf = super().to_inference_object(
            params=["beta", "phi"],
            dims=dims,
            coords=coords,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood",
            include_observed_data=True,
            **args
        )
        return inf


class NegativeBinomialLME(Model):
    """Fit count data using negative binomial model considering subject as
    a random effect.

    .. math::

        y_{ij} &\\sim \\textrm{NB}(\\mu_{ij},\\phi_j)

        \\mu_{ij} &= n_i p_{ij}

        \\textrm{alr}^{-1}(p_i) &= x_i \\beta + z_i u

    Priors:

    .. math::

        \\beta_j &\\sim \\textrm{Normal}(0, B_p), B_p \\in \\mathbb{R}_{>0}

        \\frac{1}{\\phi_j} &\\sim \\textrm{Cauchy}(0, C_s), C_s \\in
            \\mathbb{R}_{>0}

        u_j &\\sim \\textrm{Normal}(0, u_p), u_p \\in \\mathbb{R}_{>0}


    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param group_var: Variable in metadata to use as grouping
    :type group_var: str

    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame

    :param num_iter: Number of posterior sample draws, defaults to 500
    :type num_iter: int

    :param num_warmup: Number of posterior draws used for warmup, defaults to
        num_iter
    :type num_warmup: int

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float

    :param beta_prior: Standard deviation for normally distributed prior values
        of beta, defaults to 5.0
    :type beta_prior: float

    :param cauchy_scale: Scale parameter for half-Cauchy distributed prior
        values of phi, defaults to 5.0
    :type cauchy_scale: float

    :param group_var_prior: Standard deviation for normally distributed prior
        values of group_var, defaults to 1.0
    :type group_var_prior: float
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        group_var: str,
        metadata: pd.DataFrame,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
        group_var_prior: float = 1.0
    ):
        filepath = DEFAULT_MODEL_DICT["negative_binomial"]["lme"]
        super().__init__(
            table=table,
            formula=formula,
            metadata=metadata,
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
            parallelize_across="chains"
        )

        # Encode group IDs starting at 1 because Stan 1-indexes arrays
        group_var_series = metadata[group_var].loc[self.sample_names]
        samp_subj_map = group_var_series.astype("category").cat.codes + 1
        # Encoding as categories uses alphabetic sorting
        self.groups = np.sort(group_var_series.unique())

        param_dict = {
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "B_p": beta_prior,
            "phi_s": cauchy_scale,
            "S": len(group_var_series.unique()),
            "subj_ids": samp_subj_map.values,
            "u_p": group_var_prior
        }
        self.add_parameters(param_dict)

    def to_inference_object(self) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        dims = {
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
            "subj_int": ["group", "feature"],
            "log_lhood": ["tbl_sample", "feature"],
            "y_predict": ["tbl_sample", "feature"]
        }
        coords = {
            "covariate": self.colnames,
            "feature": self.feature_names,
            "tbl_sample": self.sample_names,
            "group": self.groups
        }

        # TODO: May want to allow not passing PP/LL/OD in the future
        inf = super().to_inference_object(
            params=["beta", "phi"],
            dims=dims,
            coords=coords,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood",
            alr_params=["beta", "subj_int"],
            include_observed_data=True,
        )
        return inf


class Multinomial(Model):
    """Fit count data using serial multinomial model.

    .. math::

        y_i &\\sim \\textrm{Multinomial}(\\eta_i)

        \\eta_i &= \\textrm{alr}^{-1}(x_i \\beta)

    Priors:

    .. math::

        \\beta_j \\sim \\textrm{Normal}(0, B_p), B_p \\in \\mathbb{R}_{>0}

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame

    :param num_iter: Number of posterior sample draws, defaults to 500
    :type num_iter: int

    :param num_warmup: Number of posterior draws used for warmup, defaults to
        num_iter
    :type num_warmup: int

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float

    :param beta_prior: Standard deviation for normally distributed prior values
        of beta, defaults to 5.0
    :type beta_prior: float
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
        beta_prior: float = 5.0,
    ):
        filepath = DEFAULT_MODEL_DICT["multinomial"]
        super().__init__(
            table=table,
            formula=formula,
            metadata=metadata,
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
            parallelize_across="chains"
        )

        param_dict = {
            "B_p": beta_prior,
            "depth": table.sum(axis="sample").astype(int)
        }
        self.add_parameters(param_dict)

    def to_inference_object(self) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        dims = {
            "beta": ["covariate", "feature"],
            "log_lhood": ["tbl_sample"],
            "y_predict": ["tbl_sample", "feature"]
        }
        coords = {
            "covariate": self.colnames,
            "feature": self.feature_names,
            "tbl_sample": self.sample_names
        }

        # TODO: May want to allow not passing PP/LL/OD in the future
        inf = super().to_inference_object(
            params=["beta", "phi"],
            dims=dims,
            coords=coords,
            alr_params=["beta"],
            posterior_predictive="y_predict",
            log_likelihood="log_lhood",
            include_observed_data=True,
        )
        return inf
