from typing import Sequence
import warnings

import arviz as az
import biom
from cmdstanpy import CmdStanModel, CmdStanMCMC
import dask
import numpy as np
import pandas as pd
from patsy import dmatrix

from .model_util import single_fit_to_inference, multiple_fits_to_inference


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
        # Set data back to full table
        self.dat["y"] = self.table.matrix_data.todense().T.astype(int)
        return _fits

    def to_inference_object(
        self,
        params: Sequence[str],
        coords: dict,
        dims: dict,
        concatenation_name: str = "feature",
        alr_params: Sequence[str] = None,
        include_observed_data: bool = False,
        posterior_predictive: str = None,
        log_likelihood: str = None
    ) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        Example for a simple Negative Binomial model:

        .. code-block:: python

            inf_obj = model.to_inference_object(
                params=['beta', 'phi'],
                coords={
                    'feature': model.feature_names,
                    'covariate': model.colnames
                },
                dims={
                    'beta': ['covariate', 'feature'],
                    'phi': ['feature']
                },
                alr_params=['beta']
            )

        :param params: Posterior fitted parameters to include
        :type params: Sequence[str]

        :param coords: Mapping of entries in dims to labels
        :type coords: dict

        :param dims: Dimensions of parameters in the model
        :type dims: dict

        :param concatenation_name: Name to aggregate features when combining
            multiple fits, defaults to 'feature'
        :type concatentation_name: str, optional

        :param alr_params: Parameters to convert from ALR to CLR (this will
            be ignored if the model has been parallelized across features)
        :type alr_params: Sequence[str], optional

        :param include_observed_data: Whether to include the original feature
            table values into the ``arviz`` InferenceData object, default is
            False
        :type include_observed_data: bool, optional

        :param posterior_predictive: Name of posterior predictive values from
            Stan model to include in ``arviz`` InferenceData object
        :type posterior_predictive: str, optional

        :param log_likelihood: Name of log likelihood values from Stan model
            to include in ``arviz`` InferenceData object
        :type log_likelihood: str, optional

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        args = {
            "params": params,
            "coords": coords,
            "dims": dims,
            "posterior_predictive": posterior_predictive,
            "log_likelihood": log_likelihood,
            "sample_names": self.sample_names
        }
        if isinstance(self.fit, CmdStanMCMC):
            fit_to_inference = single_fit_to_inference
            args["alr_params"] = alr_params
        elif isinstance(self.fit, Sequence):
            fit_to_inference = multiple_fits_to_inference
            args["concatenation_name"] = concatenation_name
            # TODO: Check that dims and concatenation_match

            if alr_params is not None:
                warnings.warn("ALR to CLR not performed on parallel models.",
                              UserWarning)
        else:
            raise ValueError("Unrecognized fit type!")

        inference = fit_to_inference(self.fit, **args)
        if include_observed_data:
            obs = az.from_dict(
                observed_data={"observed": self.dat["y"]},
                coords={
                    "sample": self.sample_names,
                    "feature": self.feature_names
                },
                dims={"observed": ["sample", "feature"]}
            )
            inference = az.concat(inference, obs)
        return inference

    def diagnose(self):
        """Use built-in diagnosis function of ``cmdstanpy``."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")
        if self.parallelize_across == "chains":
            return self.fit.diagnose()
        if self.parallelize_across == "features":
            return [x.diagnose() for x in self.fit]

    def summary(self):
        """Use built-in summary function of ``cmdstanpy``."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")
        if self.parallelize_across == "chains":
            return self.fit.summary()
        if self.parallelize_across == "features":
            return [x.summary() for x in self.fit]
