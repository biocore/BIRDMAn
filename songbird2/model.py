from pkgutil import get_data

import numpy as np
import pandas as pd
import pystan


MODEL_DICT = {"negative_binomial": "templates/negative_binomial.stan"}


class Model:
    def __init__(
        self,
        table: pd.DataFrame,
        dmat: pd.DataFrame,
        num_iter: int = 2000,
        chains: int = 4,
        num_jobs: int = -1,
        seed: float = None,
        model_type: str = None,
    ):
        self.num_iter = num_iter
        self.chains = chains
        self.num_jobs = num_jobs
        self.seed = seed
        self.feature_names = table.columns.tolist()
        self.sample_names = table.index.tolist()
        self.model_type = model_type

        if model_type is not None:
            stanfile_path = MODEL_DICT.get(model_type)
            stanfile = get_data(__name__, stanfile_path).decode("utf-8")
            sm = pystan.StanModel(model_code=str(stanfile))
            self.sm = sm
            print("Model compiled!")

        self.dat = {
            "N": table.shape[0],
            "D": table.shape[1],
            "p": dmat.shape[1],
            "depth": np.log(table.sum(axis=1)),
            "x": dmat.values,
            "y": table.values.astype(np.int64),
        }

    def _fit(self):
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
        super().__init__(table, dmat, num_iter, chains, num_jobs, seed,
                         model_type="negative_binomial")
        param_dict = {
            "B_p": beta_prior,
            "phi_s": cauchy_scale
        }
        self.dat.update(param_dict)
