from abc import ABC, abstractmethod
from typing import Sequence
import warnings

import arviz as az
import biom
from cmdstanpy import CmdStanModel
import pandas as pd
from patsy import dmatrix

from .model_util import full_fit_to_inference, single_feature_fit_to_inference


class BaseModel(ABC):
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
    """
    def __init__(
        self,
        table: biom.table.Table,
        model_path: str,
        num_iter: int = 500,
        num_warmup: int = None,
        chains: int = 4,
        seed: float = 42,
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

        self.dat = {
            "D": table.shape[0],  # number of features
            "N": table.shape[1],  # number of samples
        }

        self.specifications = dict()

    def create_regression(self, formula: str, metadata: pd.DataFrame):
        """Generate design matrix for count regression modeling.

        :param formula: Design formula to use in model
        :type formula: str

        :param metadata: Metadata for design matrix
        :type metadata: pd.DataFrame
        """
        self.dmat = dmatrix(formula, metadata.loc[self.sample_names],
                            return_type="dataframe")
        self.colnames = self.dmat.columns

        param_dict = {
            "p": self.dmat.shape[1],
            "x": self.dmat.values,
        }
        self.add_parameters(param_dict)

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
        sampler_args: dict = None,
        convert_to_inference: bool = False
    ):
        """Fit Stan model.

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
            parallel_chains=self.chains,
            data=self.dat,
            iter_warmup=self.num_warmup,
            iter_sampling=self.num_iter,
            seed=self.seed,
            **sampler_args
        )

        self.fit = _fit

        # If auto-conversion fails, fit will be of type CmdStanMCMC
        if convert_to_inference:
            try:
                self.fit = self.to_inference_object()
            except Exception as e:
                print(
                    "Auto conversion to InferenceData has failed! "
                    "self.fit has been saved as CmdStanMCMC instead."
                )
                print(str(e))

    @abstractmethod
    def to_inference_object(self):
        """Convert fitted model to az.InferenceData."""


class TableModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameters(
            {"y": self.table.matrix_data.todense().T.astype(int)}
        )

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
                      "log_likelihood", "alr_params"]
        }

        inference = full_fit_to_inference(self.fit, **args)

        if self.specifications["include_observed_data"]:
            obs = az.from_dict(
                observed_data={"observed": self.dat["y"]},
                coords={
                    "tbl_sample": self.sample_names,
                    "feature": self.feature_names
                },
                dims={
                    "observed": ["tbl_sample", "feature"]
                }
            )
            inference = az.concat(inference, obs)
        return inference


class SingleFeatureModel(BaseModel):
    def __init__(self, feature_id: str, **kwargs):
        if feature_id is None:
            raise ValueError("Must provide feature ID!")

        super().__init__(**kwargs)
        self.feature_id = feature_id
        values = self.table.data(
            id=feature_id,
            axis="observation",
            dense=True
        ).astype(int)
        self.add_parameters({"y": values})

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

        if "alr_params" in self.specifications:
            warnings.warn("alr_params ignored when fitting a single feature")

        args = {
            k: self.specifications.get(k)
            for k in ["params", "coords", "dims", "posterior_predictive",
                      "log_likelihood"]
        }

        inference = single_feature_fit_to_inference(self.fit, **args)

        if self.specifications["include_observed_data"]:
            obs = az.from_dict(
                observed_data={"observed": self.dat["y"]},
                coords={"tbl_sample": self.sample_names},
                dims={"observed": ["tbl_sample"]}
            )
            inference = az.concat(inference, obs)
        return inference
