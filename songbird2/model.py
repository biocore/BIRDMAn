from pkgutil import get_data

import biom
import numpy as np
import pandas as pd
from patsy import dmatrix
import pystan


def fit_model(
    table: biom.table.Table,
    metadata: pd.DataFrame,
    formula: str,
    num_iter: int = 2000,
    chains: int = 4,
    num_jobs: int = -1,
    beta_prior: float = 5.0,
    cauchy_scale: float = 5.0,
    seed: float = None,
):
    """Fit Bayesian differential abundance model.

    Parameters:
    -----------
    table: biom.table.Table
        table of feature abundances (features x samples)
    metadata: pd.DataFrame
        DataFrame of sample metadata
    formula: str
        metadata terms for which to fit parameters
    num_iter: int (default = 2000)
        number of iterations for each Markov chain
    chains: int (default = 4)
         number of Markov chains to use
    num_jobs: int (default = -1)
        number of CPUs to use in parallel for sampling (default to all
        available CPUs)
    beta_prior: float (default = 5.0)
        standard deviation for beta normal prior
    cauchy_scale: float (default = 5.0)
        scale for dispersion Cauchy prior
    seed: float (default = None)
        random seed to use for sampling
    """
    table_df = table.to_dataframe().to_dense().T
    metadata_filt = metadata.loc[table_df.index, :]
    dmat = dmatrix(formula, metadata_filt, return_type="dataframe")
    dmat_columns = dmat.columns.tolist()

    dat = {
        "N": table_df.shape[0],
        "D": table_df.shape[1],
        "p": dmat.shape[1],
        "depth": np.log(table_df.sum(axis=1).values),
        "x": dmat.values,
        "y": table_df.astype(np.int64),
        "B_p": beta_prior,
        "phi_s": cauchy_scale,
    }
    # log depth because of log parameterization of negative binomial

    # Load Stan model and run Hamiltonian MCMC sampling
    stanfile = get_data(__name__, "model.stan").decode("utf-8")
    sm = pystan.StanModel(model_code=str(stanfile))
    fit = sm.sampling(
        data=dat,
        iter=num_iter,
        chains=chains,
        n_jobs=num_jobs,
        seed=seed,
    )
    res = fit.extract(permuted=True)

    return res, dmat_columns
