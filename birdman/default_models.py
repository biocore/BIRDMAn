import os
from pkg_resources import resource_filename

import biom
import numpy as np
import pandas as pd

from .model_base import TableModel, SingleFeatureModel

TEMPLATES = resource_filename("birdman", "templates")
DEFAULT_MODEL_DICT = {
    "negative_binomial": {
        "standard": os.path.join(TEMPLATES, "negative_binomial.stan"),
        "single": os.path.join(TEMPLATES, "negative_binomial_single.stan"),
        "lme": os.path.join(TEMPLATES, "negative_binomial_lme.stan")
    },
    "multinomial": os.path.join(TEMPLATES, "multinomial.stan"),
}


class NegativeBinomial(TableModel):
    """Fit count data using negative binomial model on full table.

    .. math::

        y_{ij} &\\sim \\textrm{NB}(\\mu_{ij},\\phi_j)

        \\mu_{ij} &= n_i p_{ij}

        \\textrm{alr}(p_i) &= x_i \\beta

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
    ):
        filepath = DEFAULT_MODEL_DICT["negative_binomial"]["standard"]

        super().__init__(
            table=table,
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
        )
        self.create_regression(formula=formula, metadata=metadata)

        param_dict = {
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)

        self.specify_model(
            params=["beta", "phi"],
            dims={
                "beta": ["covariate", "feature_alr"],
                "phi": ["feature"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "covariate": self.colnames,
                "feature": self.feature_names,
                "feature_alr": self.feature_names[1:],
                "tbl_sample": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )


class NegativeBinomialSingle(SingleFeatureModel):
    """Fit count data using negative binomial model on single feature.

    .. math::

        y_{ij} &\\sim \\textrm{NB}(\\mu_{ij},\\phi_j)

        \\log(\\mu_{ij}) &= \\log(\\textrm{Depth}_i) + x_i \\beta

    Priors:

    .. math::

        \\begin{cases}
        \\beta_j \\sim \\textrm{Normal}(-5.5, B_p), & j = 0

        \\beta_j \\sim \\textrm{Normal}(0, B_p), & j > 0
        \\end{cases}

    .. math::

        B_p \\in \\mathbb{R}_{>0}

    .. math::

        \\frac{1}{\\phi_j} \\sim \\textrm{Cauchy}(0, C_s), C_s \\in
            \\mathbb{R}_{>0}


    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param feature_id: ID of feature to fit
    :type feature_id: str

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
    """
    def __init__(
        self,
        table: biom.table.Table,
        feature_id: str,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        filepath = DEFAULT_MODEL_DICT["negative_binomial"]["single"]

        super().__init__(
            table=table,
            feature_id=feature_id,
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
        )
        self.create_regression(formula=formula, metadata=metadata)

        param_dict = {
            "depth": np.log(table.sum(axis="sample")),
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)

        self.specify_model(
            params=["beta", "phi"],
            dims={
                "beta": ["covariate"],
                "log_lhood": ["tbl_sample"],
                "y_predict": ["tbl_sample"]
            },
            coords={
                "covariate": self.colnames,
                "tbl_sample": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )


class NegativeBinomialLME(TableModel):
    """Fit count data using negative binomial model considering subject as
    a random effect.

    .. math::

        y_{ij} &\\sim \\textrm{NB}(\\mu_{ij},\\phi_j)

        \\mu_{ij} &= n_i p_{ij}

        \\textrm{alr}(p_i) &= x_i \\beta + z_i u

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
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
        )
        self.create_regression(formula=formula, metadata=metadata)

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

        self.specify_model(
            params=["beta", "phi", "subj_int"],
            dims={
                "beta": ["covariate", "feature_alr"],
                "phi": ["feature"],
                "subj_int": ["group", "feature_alr"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "covariate": self.colnames,
                "feature": self.feature_names,
                "feature_alr": self.feature_names[1:],
                "tbl_sample": self.sample_names,
                "group": self.groups
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )


class Multinomial(TableModel):
    """Fit count data using serial multinomial model.

    .. math::

        y_i &\\sim \\textrm{Multinomial}(\\eta_i)

        \\eta_i &= \\textrm{alr}(x_i \\beta)

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
            model_path=filepath,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
        )
        self.create_regression(formula=formula, metadata=metadata)

        param_dict = {
            "B_p": beta_prior,
            "depth": table.sum(axis="sample").astype(int)
        }
        self.add_parameters(param_dict)

        self.specify_model(
            params=["beta"],
            dims={
                "beta": ["covariate", "feature_alr"],
                "log_lhood": ["tbl_sample"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "covariate": self.colnames,
                "feature": self.feature_names,
                "feature_alr": self.feature_names[1:],
                "tbl_sample": self.sample_names,
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )
