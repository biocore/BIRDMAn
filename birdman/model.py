from pkgutil import get_data
import warnings

import biom
import dask
import numpy as np
import pandas as pd
from patsy import dmatrix
import pystan

from .util import setup_dask_client


MODEL_DICT = {
    "negative_binomial": "templates/negative_binomial.stan",
    "multinomial": "templates/multinomial.stan",
    "negative_binomial_dask": "templates/negative_binomial_single.stan"
}


class Model:
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        model_type: str,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
    ):
        self.table = table
        self.num_iter = num_iter
        self.chains = chains
        self.num_jobs = num_jobs
        self.seed = seed
        self.formula = formula
        self.feature_names = table.ids(axis="observation")
        self.sample_names = table.ids(axis="sample")
        self.model_type = model_type
        self.sm = None

        self.dmat = dmatrix(formula, metadata.loc[self.sample_names],
                            return_type="dataframe")
        self.colnames = self.dmat.columns.tolist()

        self.dat = {
            "N": table.shape[1],
            "D": table.shape[0],
            "p": self.dmat.shape[1],
            "depth": np.log(table.sum(axis="sample")),
            "x": self.dmat.values,
            "y": table.matrix_data.todense().T.astype(int),
        }

    def compile_model(self, filepath=None):
        """Compile Stan model.

        By default if model_type is recognized the appropriate Stan file will
        be loaded and compiled.

        Parameters:
        -----------
        filepath : str
            If provided, will be loaded and compiled.
        """
        self.filepath = filepath
        if self.model_type in MODEL_DICT:
            if filepath is not None:
                warnings.warn(
                    "Ignoring provided filepath and using built-in "
                    f"{self.model_type} model instead."
                )
            stanfile_path = MODEL_DICT.get(self.model_type)
            stanfile = get_data(__name__, stanfile_path).decode("utf-8")
        elif filepath is not None:
            with open(filepath, "r") as f:
                stanfile = f.read()
        else:
            raise ValueError("Unsupported model type!")

        sm = pystan.StanModel(model_code=str(stanfile))
        self.sm = sm
        print("Model compiled successfully!")

    def add_parameters(self, param_dict=None):
        """Add parameters from dict to be passed to Stan."""
        self.dat.update(param_dict)

    def fit_model(self, **kwargs):
        """Draw posterior samples from the model."""
        if self.sm is None:
            raise ValueError("Must compile model first!")

        self.fit = self.sm.sampling(
            data=self.dat,
            iter=self.num_iter,
            chains=self.chains,
            n_jobs=self.num_jobs,
            seed=self.seed,
            **kwargs,
        )
        return self.fit


class NegativeBinomial(Model):
    """Fit count data using negative binomial model.

    Parameters:
    -----------
    beta_prior : float
        Normal prior standard deviation parameter for beta (default = 5.0)
    cauchy_scale : float
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


class NegativeBinomialDask(Model):
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: biom.table.Table,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        super().__init__(table, formula, metadata, "negative_binomial_dask",
                         num_iter, chains, seed=42, num_jobs=1)
        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)
        self.filepath = MODEL_DICT["negative_binomial_dask"]

    def fit_model(self, **kwargs):
        if self.sm is None:
            raise ValueError("Must compile model first!")

        setup_dask_client()

        @dask.delayed
        def _fit_microbe(self, values):

            dat = self.dat
            dat["y"] = values.astype(int)

            _fit = self.sm.sampling(
                data=dat,
                iter=self.num_iter,
                chains=self.chains,
                n_jobs=self.num_jobs,
                seed=self.seed,
            )
            return _fit

        _fits = []
        for v, i, d in self.table.iter(axis="observation"):
            _fit = _fit_microbe(self, v)
            _fits.append(_fit)
            print(_fit)

        fit = dask.compute(*_fits)
        print(fit)
        return fit


class Multinomial(Model):
    """Fit count data using multinomial model.

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
        super().__init__(table, formula, metadata, "negative_binomial",
                         num_iter, chains, num_jobs, seed)
        param_dict = {
            "B_p": beta_prior,
        }
        self.add_parameters(param_dict)
        self.filepath = MODEL_DICT["multinomial"]
