from pkgutil import get_data
import warnings

import numpy as np
import pandas as pd
import pystan


MODEL_DICT = {"negative_binomial": "templates/negative_binomial.stan"}


class Model:
    def __init__(
        self,
        table: pd.DataFrame,
        dmat: pd.DataFrame,
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
        self.feature_names = table.columns.tolist()
        self.sample_names = table.index.tolist()
        self.colnames = dmat.columns.tolist()
        self.model_type = model_type
        self.sm = None

        self.dat = {
            "N": table.shape[0],
            "D": table.shape[1],
            "p": dmat.shape[1],
            "depth": np.log(table.sum(axis=1)),
            "x": dmat.values,
            "y": table.values.astype(np.int64),
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
                    f"Ignoring provided filepath and using built-in "
                    "{self.model_type} model instead."
                )
            stanfile_path = MODEL_DICT.get(self.model_type)
            stanfile = get_data(__name__, stanfile_path).decode("utf-8")
        elif filepath is not None:
            with open(filepath, "r") as f:
                stanfile = f.read().decode("utf-8")
        else:
            raise ValueError("Unsupported model type!")

        sm = pystan.StanModel(model_code=str(stanfile))
        self.sm = sm
        print("Model compiled successfully!")

    def add_parameters(self, param_dict=None):
        """Add parameters from dict to be passed to Stan."""
        self.dat.update(param_dict)

    def fit_model(self):
        if self.sm is None:
            raise ValueError("Must compile model first!")

        self.fit = self.sm.sampling(
            data=self.dat,
            iter=self.num_iter,
            chains=self.chains,
            n_jobs=self.num_jobs,
            seed=self.seed,
        )
        return self.fit


class NegativeBinomial(Model):
    def __init__(
        self,
        table: pd.DataFrame,
        dmat: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        super().__init__(table, dmat, "negative_binomial", num_iter, chains,
                         num_jobs, seed)
        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.add_parameters(param_dict)
        self.filepath = MODEL_DICT["negative_binomial"]
