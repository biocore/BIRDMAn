from pkgutil import get_data
import warnings

import biom
import dask
import numpy as np
import pandas as pd
from patsy import dmatrix
import stan

DEFAULT_MODEL_DICT = {
    "negative_binomial": "templates/negative_binomial.stan",
    "multinomial": "templates/multinomial.stan",
    "negative_binomial_dask": "templates/negative_binomial_single.stan"
}


class Model:
    """Base Stan model."""
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        model_type: str,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = 42,
    ):
        self.table = table
        self.num_iter = num_iter
        self.chains = chains
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
            "N": table.shape[1],                        # number of samples
            "p": self.dmat.shape[1],                    # number of covariates
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "x": self.dmat.values,                      # design matrix
        }

    def load_stancode(self, filepath=None):
        """
        By default if model_type is recognized the appropriate Stan file will
        be loaded.

        Parameters:
        -----------
        filepath : str
            If provided, will be loaded and compiled.
        """
        self.filepath = filepath
        if self.model_type in DEFAULT_MODEL_DICT:
            if filepath is not None:
                warnings.warn(
                    "Ignoring provided filepath and using built-in "
                    f"{self.model_type} model instead."
                )
            stanfile_path = DEFAULT_MODEL_DICT.get(self.model_type)
            stanfile = get_data(__name__, stanfile_path).decode("utf-8")
        elif filepath is not None:
            with open(filepath, "r") as f:
                stanfile = f.read()
        else:
            raise ValueError("Unsupported model type!")

        self.stancode = stanfile

    def add_parameters(self, param_dict=None):
        """Add parameters from dict to be passed to Stan."""
        self.dat.update(param_dict)


class SerialModel(Model):
    """Stan model parallelizing MCMC chains."""
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        model_type: str,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = 42,
    ):
        super().__init__(table, formula, metadata, model_type, num_iter,
                         chains, seed)
        param_dict = {
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0]
        }
        self.add_parameters(param_dict)

    def fit_model(self, **kwargs):
        """Draw posterior samples from the model."""
        sm = stan.build(self.stancode, data=self.dat, random_seed=self.seed)

        self.fit = sm.sample(
            num_chains=self.chains,
            num_samples=self.num_iter,
        )
        return self.fit


class ParallelModel(Model):
    """Stan model parallelized across each microbe."""
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: biom.table.Table,
        model_type: str,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = None,
    ):
        super().__init__(table, formula, metadata, model_type, num_iter,
                         chains, seed)

    def fit_model(self, **kwargs):
        @dask.delayed
        def _fit_microbe(self, values):

            dat = self.dat
            dat["y"] = values.astype(int)
            sm = stan.build(self.stancode, data=dat,
                            random_seed=self.seed)

            _fit = sm.sample(
                num_chains=self.chains,
                num_samples=self.num_iter,
            )
            return _fit

        _fits = []
        for v, i, d in self.table.iter(axis="observation"):
            _fit = _fit_microbe(self, v)
            _fits.append(_fit)

        _fits = dask.compute(*_fits)  # D-tuple
        self.fit = _fits
        return _fits
