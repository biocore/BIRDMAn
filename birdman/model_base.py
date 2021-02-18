from typing import Sequence
import warnings

import arviz as az
import biom
from cmdstanpy import CmdStanModel, CmdStanMCMC
import dask
import numpy as np
import pandas as pd
from patsy import dmatrix

from .model_util import single_fit_to_xarray, multiple_fits_to_xarray


class Model:
    """Base Stan model.

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param metadata: Metadata file for design matrix
    :type metadata: pd.DataFrame

    :param model_path: Filepath to Stan model
    :type model_path: str

    :param num_iter: Number of posterior draws (used for both warmup and
        sampling), defaults to 2000
    :type num_iter: int, optional

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int, optional

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float, optional

    :param parallelize_across: Whether to parallelize across features or chains
        , defaults to 'chains'
    :type parallelize_across: str, optional
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        model_path: str,
        num_iter: int = 2000,
        chains: int = 4,
        seed: float = 42,
        parallelize_across: str = "chains"
    ):
        self.table = table
        self.num_iter = num_iter
        self.chains = chains
        self.seed = seed
        self.formula = formula
        self.feature_names = table.ids(axis="observation")
        self.sample_names = table.ids(axis="sample")
        self.model_path = model_path
        self.sm = None
        self.fit = None
        self.parallelize_across = parallelize_across

        self.dmat = dmatrix(formula, metadata.loc[self.sample_names],
                            return_type="dataframe")
        self.colnames = self.dmat.columns

        self.dat = {
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0],
            "N": table.shape[1],                        # number of samples
            "p": self.dmat.shape[1],                    # number of covariates
            "depth": np.log(table.sum(axis="sample")),  # sampling depths
            "x": self.dmat.values,                      # design matrix
        }

    def compile_model(self):
        """Compile Stan model."""
        self.sm = CmdStanModel(stan_file=self.model_path)

    def add_parameters(self, param_dict=None):
        """Add parameters from dict to be passed to Stan."""
        self.dat.update(param_dict)

    def fit_model(self, **kwargs):
        """Fit model according to parallelization configuration."""
        if self.parallelize_across == "features":
            self.fit = self._fit_parallel(**kwargs)
        elif self.parallelize_across == "chains":
            self.fit = self._fit_serial(**kwargs)
        else:
            raise ValueError("parallelize_across must be features or chains!")

    def _fit_serial(self, **kwargs):
        """Fit model by parallelizing across chains."""
        fit = self.sm.sample(
            chains=self.chains,
            parallel_chains=self.chains,  # run all chains in parallel
            data=self.dat,
            iter_warmup=self.num_iter,    # use same num iter for warmup
            iter_sampling=self.num_iter,
            seed=self.seed,
            **kwargs
        )
        return fit

    def _fit_parallel(self, **kwargs):
        """Fit model by parallelizing across features."""
        @dask.delayed
        def _fit_single(self, values):
            dat = self.dat
            dat["y"] = values.astype(int)
            _fit = self.sm.sample(
                chains=self.chains,
                parallel_chains=1,            # run all chains in serial
                data=dat,
                iter_warmup=self.num_iter,    # use same num iter for warmup
                iter_sampling=self.num_iter,
                seed=self.seed,
                **kwargs
            )
            return _fit

        _fits = []
        for v, i, d in self.table.iter(axis="observation"):
            _fit = _fit_single(self, v)
            _fits.append(_fit)

        _fits = dask.compute(*_fits)
        return _fits

    def to_inference_object(
            self,
            params_to_include: Sequence,
            feature_names: Sequence = None,
            covariate_names: Sequence = None,
            alr_params: Sequence = None,
    ) -> az.InferenceData:
        """Convert fitted Stan model into arviz InferenceData object.

        .. note:: We don't use the arviz from_cmdstanpy function because it
            does not transform the ALR coordinates and returns all parameters
            (including irrelevant intermediates).

        :param params_to_include: Names of parameters to keep
        :type params_to_include: Sequence[str]

        :param feature_names: Names of features, defaults to biom table
            observation ids
        :type feature_names: Sequence[str]

        :param covariate_names: Names of covariates in design matrix, defaults
            to columns of dmat
        :type covariate_names: Sequence[str]

        :param alr_params: Parameters to convert from ALR to CLR
        :type alr_params: Sequence[str]

        :returns: arviz InferenceData object with selected values/coordinates
        :rtype: az.InferenceData
        """
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        if feature_names is None:
            feature_names = self.table.ids(axis="observation")
        if covariate_names is None:
            covariate_names = self.dmat.columns.tolist()

        if isinstance(self.fit, CmdStanMCMC):
            ds = single_fit_to_xarray(
                fit=self.fit,
                params=params_to_include,
                feature_names=feature_names,
                covariate_names=covariate_names,
                alr_params=alr_params,
            )
        elif isinstance(self.fit, Sequence):
            if alr_params is not None:
                warnings.warn("ALR to CLR not performed on parallel models.",
                              UserWarning)
            ds = multiple_fits_to_xarray(
                fits=self.fit,
                params=params_to_include,
                feature_names=feature_names,
                covariate_names=covariate_names
            )
        else:
            raise ValueError("Unrecognized fit type!")
        return az.convert_to_inference_data(ds)
