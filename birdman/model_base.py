from pkgutil import get_data
import warnings

import biom
import dask
import numpy as np
import pandas as pd
from patsy import dmatrix
import pystan

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
            "N": table.shape[1],  # number of samples
            "p": self.dmat.shape[1],  # number of covariates
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "x": self.dmat.values,  # design matrix
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

        sm = pystan.StanModel(model_code=str(stanfile))
        self.sm = sm
        print("Model compiled successfully!")

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
        num_jobs: int = -1,
        seed: float = None,
    ):
        super().__init__(table, formula, metadata, model_type, num_iter, chains,
                         num_jobs, seed)
        param_dict = {
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0]
        }
        self.add_parameters(param_dict)

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

    def collapse_param(self, param: str, convert_alr_to_clr: bool = False):
        """Compute mean and stdev for parameter from posterior samples."""
        dfs = []
        res = _extract_params(self.fit)

        param_data = res[param]

        # TODO: figure out how to vectorize this
        if param_data.ndim == 3:  # matrix parameter
            for i, colname in enumerate(self.colnames):
                if convert_alr_to_clr:  # for beta parameters
                    data = alr_to_clr(param_data[:, i, :])
                else:
                    data = param_data
                mean = pd.DataFrame(data.mean(axis=0))
                std = pd.DataFrame(data.std(axis=0))
                df = pd.concat([mean, std], axis=1)
                df.columns = [f"{colname}_{x}" for x in ["mean", "std"]]
                dfs.append(df)
            param_df = pd.concat(dfs, axis=1)
        elif param_data.ndim == 2:  # vector parameter
            if convert_alr_to_clr:
                data = alr_to_clr(param_data)
            else:
                data = param_data
            mean = pd.DataFrame(data.mean(axis=0))
            std = pd.DataFrame(data.std(axis=0))
            param_df = pd.concat([mean, std], axis=1)
            param_df.columns = [f"{param}_{x}" for x in ["mean", "std"]]
        else:
            raise ValueError("Parameter must be matrix or vector type!")
        return param_df

    def _extract_params(self):
        """Helper function so that this can be mocked for testing."""
        return self.fit.extract(permuted=True)


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
        num_jobs: int = -1,
        seed: float = None,
        beta_prior: float = 5.0,
        cauchy_scale: float = 5.0,
    ):
        super().__init__(table, formula, metadata, model_type, num_iter,
                         chains, seed=42, num_jobs=1)

    def fit_model(self, **kwargs):
        if self.sm is None:
            raise ValueError("Must compile model first!")

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

        _fits = dask.compute(*_fits)  # D-tuple
        self.fit = _fits
        return _fits
