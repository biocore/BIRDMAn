__all__ = ["NegativeBinomial", "Multinomial"]

import os
from pkg_resources import resource_filename

import biom
import numpy as np
import pandas as pd
import xarray as xr

from .model_base import Model
from .util import convert_beta_coordinates

TEMPLATES = resource_filename("birdman", "templates")

DEFAULT_MODEL_DICT = {
    "negative_binomial": {
        "chains": os.path.join(TEMPLATES, "negative_binomial.stan"),
        "features": os.path.join(TEMPLATES, "negative_binomial_single.stan")
    },
    "multinomial": "templates/multinomial.stan",
}


class NegativeBinomial(Model):
    """Fit count data using serial negative binomial model.

    Parameters:
    -----------
    beta_prior : float
        Normal prior standard deviation parameter for beta (default = 5.0)
    cauchy_scale: float
        Cauchy prior scale parameter for phi (default = 5.0)
    parallelize_across: str
        Whether to parallelize across microbes or chains (default = chains)
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
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0],
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


class Multinomial(Model):
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
