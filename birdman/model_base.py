import biom
import numpy as np
import pandas as pd
from patsy import dmatrix
from cmdstanpy import CmdStanModel


class Model:
    """Base Stan model."""
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        model_path: str,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = 42,
    ):
        self.table = table
        self.num_iter = num_iter
        self.chains = chains
        self.seed = seed
        self.formula = formula
        self.feature_names = table.ids(axis="observation").tolist()
        self.sample_names = table.ids(axis="sample").tolist()
        self.model_path = model_path
        self.sm = None
        self.fit = None

        self.dmat = dmatrix(formula, metadata.loc[self.sample_names],
                            return_type="dataframe")
        self.colnames = self.dmat.columns.tolist()

        self.dat = {
            "N": table.shape[1],                        # number of samples
            "p": self.dmat.shape[1],                    # number of covariates
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "x": self.dmat.values,                      # design matrix
        }

    def compile_model(self):
        self.sm = CmdStanModel(stan_file=self.model_path)

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

    def fit_model(self):
        self.fit = self.sm.sample(
            chains=self.chains,
            parallel_chains=self.chains,  # run all chains in parallel
            data=self.dat,
            iter_warmup=self.num_iter,    # use same num iter for warmup
            iter_sampling=self.num_iter,
            seed=self.seed,
        )
        return self.fit
