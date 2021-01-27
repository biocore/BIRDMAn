__all__ = ["NegativeBinomial", "Multinomial", "NegativeBinomialDask"]

import biom
import pandas as pd

from .model_base import SerialModel, ParallelModel

MODEL_DICT = {
    "negative_binomial": "templates/negative_binomial.stan",
    "multinomial": "templates/multinomial.stan",
    "negative_binomial_dask": "templates/negative_binomial_single.stan"
}

class NegativeBinomial(SerialModel):
    """Fit count data using serial negative binomial model.

    Parameters:
    -----------
    beta_prior : float
        Normal prior standard deviation parameter for beta (default = 5.0)
    cauchy_scale: float
        Cauchy prior scale parameter for phi (default = 5.0)
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        super().__init__(table, formula, metadata, "negative_binomial",
                         num_iter, chains, num_jobs, seed)
        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)
        self.filepath = MODEL_DICT["negative_binomial"]


class Multinomial(SerialModel):
    """Fit count data using serial multinomial model.

    Parameters:
    -----------
    beta_prior : float
        Normal prior standard deviation parameter for beta (default = 5.0)
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
        beta_prior: float = 5.0,
    ):
        super().__init__(table, formula, metadata, "multinomial",
                         num_iter, chains, num_jobs, seed)
        param_dict = {
            "B_p": beta_prior,
        }
        self.add_parameters(param_dict)
        self.filepath = MODEL_DICT["multinomial"]


class NegativeBinomialDask(ParallelModel):
    """Fit count data using dask-parallelized negative binomial model.

    Parameters:
    -----------
    beta_prior : float
        Normal prior standard deviation parameter for beta (default = 5.0)
    cauchy_scale: float
        Cauchy prior scale parameter for phi (default = 5.0)
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        super().__init__(table, formula, metadata, "negative_binomial_dask",
                         num_iter, chains, num_jobs, seed)
        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)
        self.filepath = MODEL_DICT["negative_binomial_dask"]
