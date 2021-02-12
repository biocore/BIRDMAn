__all__ = ["NegativeBinomial", "Multinomial"]

import os
from pkg_resources import resource_filename

import biom
import numpy as np
import pandas as pd
import xarray as xr

from .model_base import SerialModel
from .util import convert_beta_coordinates

TEMPLATES = resource_filename("birdman", "templates")

DEFAULT_MODEL_DICT = {
    "negative_binomial": os.path.join(TEMPLATES, "negative_binomial.stan"),
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
        seed: float = None,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        filepath = DEFAULT_MODEL_DICT["negative_binomial"]
        super().__init__(table, formula, metadata, filepath, num_iter, chains,
                         seed)
        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)

    def to_xarray(self) -> xr.Dataset:
        """Convert fitted parameters to xarray object."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        beta_clr = convert_beta_coordinates(self.fit["beta"])
        ds = xr.Dataset(
            data_vars=dict(
                beta=(["covariate", "feature", "draw"], beta_clr),
                phi=(["feature", "draw"], self.fit["phi"])
            ),
            coords=dict(
                covariate=self.dmat.columns,
                feature=self.table.ids(axis="observation"),
                draw=np.arange(self.num_iter*self.chains)
            )
        )
        return ds


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
        seed: float = None,
        beta_prior: float = 5.0,
    ):
        super().__init__(table, formula, metadata, "multinomial",
                         num_iter, chains, seed)
        param_dict = {
            "B_p": beta_prior,
        }
        self.add_parameters(param_dict)
        self.filepath = DEFAULT_MODEL_DICT["multinomial"]
        self.load_stancode()

    def to_xarray(self) -> xr.Dataset:
        """Convert fitted parameters to xarray object."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        beta_clr = convert_beta_coordinates(self.fit["beta"])
        ds = xr.Dataset(
            data_vars=dict(
                beta=(["covariate", "feature", "draw"], beta_clr),
            ),
            coords=dict(
                covariates=self.dmat.columns,
                features=self.table.ids(axis="observation"),
                draws=np.arange(self.num_iter*self.chains)
            )
        )
        return ds
