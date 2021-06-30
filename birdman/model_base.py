from typing import Sequence, Union

import arviz as az
import biom
from cmdstanpy import CmdStanModel, CmdStanMCMC
import numpy as np
import pandas as pd
from patsy import dmatrix

from .model_util import single_fit_to_inference, single_feature_to_inf


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

    :param single_feature: Whether this model is for a single feature or a
        full count table, defaults to False
    :type single_feature: bool
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
        single_feature: bool = False
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
        self.single_feature = single_feature

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

        :param alr_params: Parameters to convert from ALR to CLR
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
        feature_id: str = None,
        sampler_args: dict = None,
        convert_to_inference: bool = False,
    ) -> None:
        """Perform MCMC sampling.

        :param feature_id: ID of table feature to fit
        :type feature_id: str

        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict

        :param convert_to_inference: Whether to automatically convert the
            fitted model to an az.InferenceData object (defaults to False)
        :type convert_to_inference: bool
        """
        if self.sm is None:
            raise ValueError("Model must be compiled first!")

        args = {
            "sampler_args": sampler_args,
            "convert_to_inference": convert_to_inference
        }

        if not self.single_feature:
            fit_function = self._fit_serial
        else:
            fit_function = self._fit_single
            values = self.table.data(id=feature_id, axis="observation")
            args["values"] = values

        fit_function(**args)

    def _fit_serial(
        self,
        sampler_args: dict = None,
        convert_to_inference: bool = False,
    ) -> Union[CmdStanMCMC, az.InferenceData]:
        """Fit model by parallelizing across chains.

        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict

        :param convert_to_inference: Whether to automatically convert to
            inference given model specifications, defaults to False
        :type convert_to_inference: bool
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

    def _fit_single(
        self,
        values: np.ndarray,
        sampler_args: dict = None,
        convert_to_inference: bool = False,
    ) -> Union[CmdStanMCMC, az.InferenceData]:
        """Fit single feature model.

        :param values: Counts in order of sample order
        :type values: np.ndarray

        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict

        :param convert_to_inference: Whether to automatically convert to
            inference given model specifications, defaults to False
        :type convert_to_inference: bool
        """
        if sampler_args is None:
            sampler_args = dict()

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

            _fit = single_feature_to_inf(
                fit=_fit,
                coords=self.specifications.get("coords"),
                dims=self.specifications.get("dims"),
                vars_to_drop=vars_to_drop,
                posterior_predictive=self.specifications.get(
                    "posterior_predictive"
                ),
                log_likelihood=self.specifications.get("log_likelihood")
            )
        self.fit = _fit

    def to_inference_object(self) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        # if already Inference, just return
        if isinstance(self.fit, az.InferenceData):
            return self.fit

        args = {
            k: self.specifications.get(k)
            for k in ["params", "coords", "dims", "posterior_predictive",
                      "log_likelihood"]
        }
        if not self.single_feature:
            fit_to_inference = single_fit_to_inference
            args["alr_params"] = self.specifications["alr_params"]
        else:
            fit_to_inference = single_feature_to_inf

        inference = fit_to_inference(self.fit, **args)
        if self.specifications["include_observed_data"]:
            coords = {"tbl_sample": self.sample_names}
            dims = {"observed": ["tbl_sample"]}
            if not self.single_feature:
                coords["feature"] = self.feature_names
                dims["observed"].append("feature")
            obs = az.from_dict(
                observed_data={"observed": self.dat["y"]},
                coords=coords,
                dims=dims
            )
            inference = az.concat(inference, obs)
        return inference

    def diagnose(self):
        """Use built-in diagnosis function of ``cmdstanpy``."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")
        return self.fit.diagnose()

    def summary(self):
        """Use built-in summary function of ``cmdstanpy``."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")
        return self.fit.summary()


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

    :param single_feature: Whether this model is for a single feature or a
        full count table, defaults to False
    :type single_feature: bool
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
        single_feature: bool = False
    ):
        super().__init__(
            table=table,
            metadata=metadata,
            model_path=model_path,
            num_iter=num_iter,
            num_warmup=num_warmup,
            chains=chains,
            seed=seed,
            single_feature=single_feature
        )

        self.dmat = dmatrix(formula, metadata.loc[self.sample_names],
                            return_type="dataframe")
        self.colnames = self.dmat.columns

        param_dict = {
            "p": self.dmat.shape[1],
            "x": self.dmat.values,
        }
        self.add_parameters(param_dict)
