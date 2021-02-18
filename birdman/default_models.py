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

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param metadata: Metadata file for design matrix
    :type metadata: pd.DataFrame

    :param num_iter: Number of posterior draws (used for both warmup and
        sampling), defaults to 2000
    :type num_iter: int, optional

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int, optional

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float, optional

    :param beta_prior: Standard deviation for normally distributed prior values
        of beta, defaults to 5.0
    :type beta_prior: float, optional

    :param cauchy_scale: Scale parameter for half-Cauchy distributed prior
        values of phi, defaults to 5.0
    :type cauchy_scale: float, optional

    :param parallelize_across: Whether to parallelize across features or chains
        , defaults to 'chains'
    :type parallelize_across: str, optional
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = None,
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

    :param metadata: Metadata file for design matrix
    :type metadata: pd.DataFrame

    :param num_iter: Number of posterior draws (used for both warmup and
        sampling), defaults to 2000
    :type num_iter: int, optional

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int, optional

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float, optional

    :param beta_prior: Standard deviation for normally distributed prior values
        of beta, defaults to 5.0
    :type beta_prior: float, optional
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = None,
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
