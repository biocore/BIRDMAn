from pkgutil import get_data
import warnings

import biom
import numpy as np
import pandas as pd
from patsy import dmatrix
import pystan


MODEL_DICT = {
    "negative_binomial": "templates/negative_binomial.stan",
    "multinomial": "templates/multinomial.stan"
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
        self.num_iter = num_iter
        self.chains = chains
        self.num_jobs = num_jobs
        self.seed = seed
        self.formula = formula
        self.feature_names = table.ids(axis="observation")
        self.sample_names = table.ids(axis="sample")
        self.model_type = model_type
        self.sm = None

        dmat = dmatrix(formula, metadata, return_type="dataframe")
        self.colnames = dmat.columns.tolist()

        self.dat = {
            "N": table.shape[1],
            "D": table.shape[0],
            "p": dmat.shape[1],
            "depth": np.log(table.sum(axis="sample")),
            "x": dmat.values,
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
        table: pd.DataFrame,
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


class Multinomial(Model):
    """Fit count data using multinomial model.

    Parameters:
    -----------
    beta_prior : float
        Normal prior standard deviation parameter for beta (default = 5.0)
    """
    def __init__(
        self,
        table: pd.DataFrame,
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
