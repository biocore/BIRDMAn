from typing import List, Sequence, Union
import warnings

import arviz as az
import biom
from cmdstanpy import CmdStanModel, CmdStanMCMC
import dask
import numpy as np
import pandas as pd
from patsy import dmatrix

from .model_util import (single_fit_to_inference, multiple_fits_to_inference,
                         _single_feature_to_inf, concatenate_inferences)


class BaseModel:
    """Base BIRDMAn model.

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame

    :param model_path: Filepath to Stan model
    :type model_path: str

    :param num_iter: Number of posterior sample draws, defaults to 500
    :type num_iter: int

    :param num_warmup: Number of posterior draws used for warmup, defaults to
        num_iter
    :type num_warmup: int

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float

    :param parallelize_across: Whether to parallelize across features or
        chains, defaults to 'chains'
    :type parallelize_across: str
    """
    def __init__(
        self,
        table: biom.table.Table,
        metadata: pd.DataFrame,
        model_path: str,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
        parallelize_across: str = "chains"
    ):
        self.table = table
        self.num_iter = num_iter
        if num_warmup is None:
            self.num_warmup = num_iter
        else:
            self.num_warmup = num_warmup
        self.chains = chains
        self.seed = seed
        self.feature_names = table.ids(axis="observation")
        self.sample_names = table.ids(axis="sample")
        self.model_path = model_path
        self.sm = None
        self.fit = None
        self.parallelize_across = parallelize_across

        self.dat = {
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0],  # number of features
            "N": table.shape[1],  # number of samples
        }

        self.specifications = dict()

    def compile_model(self) -> None:
        """Compile Stan model."""
        self.sm = CmdStanModel(stan_file=self.model_path)

    def specify_model(
        self,
        params: Sequence[str],
        coords: dict,
        dims: dict,
        concatenation_name: str = "feature",
        alr_params: Sequence[str] = None,
        include_observed_data: bool = False,
        posterior_predictive: str = None,
        log_likelihood: str = None,
    ) -> None:
        """Specify coordinates and dimensions of model.

        :param params: Posterior fitted parameters to include
        :type params: Sequence[str]

        :param coords: Mapping of entries in dims to labels
        :type coords: dict

        :param dims: Dimensions of parameters in the model
        :type dims: dict

        :param concatenation_name: Name to aggregate features when combining
            multiple fits, defaults to 'feature'
        :type concatentation_name: str

        :param alr_params: Parameters to convert from ALR to CLR (this will
            be ignored if the model has been parallelized across features)
        :type alr_params: Sequence[str], optional

        :param include_observed_data: Whether to include the original feature
            table values into the ``arviz`` InferenceData object, default is
            False
        :type include_observed_data: bool

        :param posterior_predictive: Name of posterior predictive values from
            Stan model to include in ``arviz`` InferenceData object
        :type posterior_predictive: str, optional

        :param log_likelihood: Name of log likelihood values from Stan model
            to include in ``arviz`` InferenceData object
        :type log_likelihood: str, optional
        """
        self.specifications["params"] = params
        self.specifications["coords"] = coords
        self.specifications["dims"] = dims
        self.specifications["alr_params"] = alr_params
        self.specifications["concatenation_name"] = concatenation_name
        self.specifications["include_observed_data"] = include_observed_data
        self.specifications["posterior_predictive"] = posterior_predictive
        self.specifications["log_likelihood"] = log_likelihood

    def add_parameters(self, param_dict: dict = None) -> None:
        """Add parameters from dict to be passed to Stan."""
        if param_dict is None:
            param_dict = dict()
        self.dat.update(param_dict)

    def fit_model(
        self,
        sampler_args: dict = None,
        convert_to_inference: bool = False,
    ) -> None:
        """Fit model according to parallelization configuration.

        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict

        :param convert_to_inference: Whether to automatically convert the
            fitted model to an az.InferenceData object (defaults to False)
        :type convert_to_inference: bool
        """
        if self.sm is None:
            raise ValueError("Model must be compiled first!")

        if self.parallelize_across == "features":
            cn = self.specifications["concatenation_name"]
            self.specifications["dims"] = {
                k: [dim for dim in v if dim != cn]
                for k, v in self.specifications["dims"].items()

            }
            self._fit_parallel(
                sampler_args=sampler_args,
                convert_to_inference=convert_to_inference
            )
        elif self.parallelize_across == "chains":
            self._fit_serial(
                sampler_args=sampler_args,
                convert_to_inference=convert_to_inference
            )
        else:
            raise ValueError("parallelize_across must be features or chains!")

    def _fit_serial(
        self,
        sampler_args: dict = None,
        convert_to_inference: bool = False,
    ) -> CmdStanMCMC:
        """Fit model by parallelizing across chains.

        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict
        """
        if sampler_args is None:
            sampler_args = dict()

        _fit = self.sm.sample(
            chains=self.chains,
            parallel_chains=self.chains,  # run all chains in parallel
            data=self.dat,
            iter_warmup=self.num_warmup,
            iter_sampling=self.num_iter,
            seed=self.seed,
            **sampler_args
        )
        if convert_to_inference:
            _fit = single_fit_to_inference(
                fit=_fit,
                params=self.specifications.get("params"),
                coords=self.specifications.get("coords"),
                dims=self.specifications.get("dims"),
                alr_params=self.specifications.get("alr_params"),
                posterior_predictive=self.specifications.get(
                    "posterior_predictive"
                ),
                log_likelihood=self.specifications.get("log_likelihood")
            )
        self.fit = _fit

    def _fit_parallel(
        self,
        convert_to_inference: bool = False,
        sampler_args: dict = None,
    ) -> Union[List[CmdStanMCMC], List[az.InferenceData]]:
        """Fit model by parallelizing across features.

        :param convert_to_inference: Whether to create individual
            InferenceData objects for individual feature fits, defaults to
            False
        :type convert_to_inference: bool

        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict
        """
        if sampler_args is None:
            sampler_args = dict()

        _fits = []
        for v, i, d in self.table.iter(axis="observation"):
            _fit = dask.delayed(self._fit_single)(
                v,
                sampler_args,
                convert_to_inference,
            )
            _fits.append(_fit)

        fit_futures = dask.persist(*_fits)
        all_fits = dask.compute(fit_futures)[0]
        # Set data back to full table
        self.dat["y"] = self.table.matrix_data.todense().T.astype(int)
        self.fit = all_fits

    def _fit_single(
        self,
        values: np.ndarray,
        sampler_args: dict = None,
        convert_to_inference: bool = False,
    ) -> Union[CmdStanMCMC, az.InferenceData]:
        dat = self.dat
        dat["y"] = values.astype(int)
        _fit = self.sm.sample(
            chains=self.chains,
            data=dat,
            iter_warmup=self.num_warmup,
            iter_sampling=self.num_iter,
            seed=self.seed,
            **sampler_args
        )

        if convert_to_inference:
            all_vars = _fit.stan_variables().keys()
            vars_to_drop = set(all_vars).difference(
                self.specifications["params"]
            )
            if self.specifications.get("posterior_predictive") is not None:
                vars_to_drop.remove(
                    self.specifications["posterior_predictive"]
                )
            if self.specifications.get("log_likelihood") is not None:
                vars_to_drop.remove(self.specifications["log_likelihood"])

            _fit = _single_feature_to_inf(
                fit=_fit,
                coords=self.specifications.get("coords"),
                dims=self.specifications.get("dims"),
                vars_to_drop=vars_to_drop,
                posterior_predictive=self.specifications.get(
                    "posterior_predictive"
                ),
                log_likelihood=self.specifications.get("log_likelihood")
            )
        return _fit

    def to_inference_object(
        self,
        combine_individual_fits: bool = True,
    ) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :param combine_individual_fits: Whether to combine the results of
            parallelized feature fits, defaults to True
        :type combine_individual_fits: bool

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        # if already Inference, just return
        if isinstance(self.fit, az.InferenceData):
            return self.fit
        # if sequence of Inferences, concatenate if specified
        if isinstance(self.fit, list) or isinstance(self.fit, tuple):
            if isinstance(self.fit[0], az.InferenceData):
                if combine_individual_fits:
                    cat_name = self.specifications["concatenation_name"]
                    return concatenate_inferences(
                        self.fit,
                        coords=self.specifications["coords"],
                        concatenation_name=cat_name
                    )
                else:
                    return self.fit

        args = {
            k: self.specifications.get(k)
            for k in ["params", "coords", "dims", "posterior_predictive",
                      "log_likelihood"]
        }
        if isinstance(self.fit, CmdStanMCMC):
            fit_to_inference = single_fit_to_inference
            args["alr_params"] = self.specifications["alr_params"]
        elif isinstance(self.fit, Sequence):
            fit_to_inference = multiple_fits_to_inference
            if combine_individual_fits:
                args["concatenation_name"] = self.specifications.get(
                    "concatenation_name", "feature"
                )
                args["concatenate"] = True
            else:
                args["concatenate"] = False
            # TODO: Check that dims and concatenation_match

            if self.specifications.get("alr_params") is not None:
                warnings.warn("ALR to CLR not performed on parallel models.",
                              UserWarning)
        else:
            raise ValueError("Unrecognized fit type!")

        inference = fit_to_inference(self.fit, **args)
        if self.specifications["include_observed_data"]:
            # Can't include observed data in individual fits
            include_obs_fail = (
                not combine_individual_fits
                and self.parallelize_across == "features"
            )
            if include_obs_fail:
                warnings.warn(
                    "Cannot include observed data in un-concatenated"
                    "fits!"
                )
            else:
                obs = az.from_dict(
                    observed_data={"observed": self.dat["y"]},
                    coords={
                        "tbl_sample": self.sample_names,
                        "feature": self.feature_names
                    },
                    dims={"observed": ["tbl_sample", "feature"]}
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


class RegressionModel(BaseModel):
    """Base BIRDMAn regression model.

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param formula: Design formula to use in model
    :type formula: str

    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame

    :param model_path: Filepath to Stan model
    :type model_path: str

    :param num_iter: Number of posterior sample draws, defaults to 500
    :type num_iter: int

    :param num_warmup: Number of posterior draws used for warmup, defaults to
        num_iter
    :type num_warmup: int

    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int

    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float

    :param parallelize_across: Whether to parallelize across features or
        chains, defaults to 'chains'
    :type parallelize_across: str
    """
    def __init__(
        self,
        table: biom.table.Table,
        formula: str,
        metadata: pd.DataFrame,
        model_path: str,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
        parallelize_across: str = "chains"
    ):
        super().__init__(
            table=table,
            metadata=metadata,
            model_path=model_path,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
            parallelize_across=parallelize_across
        )

        self.dmat = dmatrix(formula, metadata.loc[self.sample_names],
                            return_type="dataframe")
        self.colnames = self.dmat.columns

        param_dict = {
            "p": self.dmat.shape[1],
            "x": self.dmat.values,
        }
        self.add_parameters(param_dict)
