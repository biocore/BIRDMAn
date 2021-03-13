import os
from pkg_resources import resource_filename

import biom
import pandas as pd

from .model_base import Model

TEMPLATES = resource_filename("birdman", "templates")
DEFAULT_MODEL_DICT = {
    "negative_binomial": {
        "chains": os.path.join(TEMPLATES, "negative_binomial.stan"),
        "features": os.path.join(TEMPLATES, "negative_binomial_single.stan")
    },
    "multinomial": "templates/multinomial.stan",
}


class NegativeBinomial(Model):
    """Fit count data using negative binomial model.

    .. math::

        y_{ij} &\\sim \\textrm{NB}(\\mu_{ij},\\phi_j)

        \\mu_{ij} &= n_i \\times p_{ij}

        \\textrm{alr}^{-1}(p_i) &= x_i \\cdot \\beta

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

    :param num_iter: Number of posterior draws (used for both warmup and
        sampling), defaults to 500
    :type num_iter: int

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

    :param parallelize_across: Whether to parallelize across features or chains
        , defaults to 'chains'
    :type parallelize_across: str
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 500,
        chains: int = 4,
        seed: float = 42,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
        parallelize_across: str = "chains",
    ):
        filepath = DEFAULT_MODEL_DICT["negative_binomial"][parallelize_across]
        super().__init__(table, formula, metadata, filepath, num_iter, chains,
                         seed, parallelize_across)

        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)


class Multinomial(Model):
    """Fit count data using serial multinomial model.

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame

    :param num_iter: Number of posterior draws (used for both warmup and
        sampling), defaults to 500
    :type num_iter: int

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
        chains: int = 4,
        seed: float = 42,
        beta_prior: float = 5.0,
    ):
        super().__init__(table, formula, metadata, "multinomial",
                         num_iter, chains, seed, parallelize_across="chains")
        param_dict = {
            "B_p": beta_prior,
        }
        self.add_parameters(param_dict)
        self.filepath = DEFAULT_MODEL_DICT["multinomial"]
        self.load_stancode()
