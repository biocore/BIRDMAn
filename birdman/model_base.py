from abc import ABC, abstractmethod
from math import ceil
from typing import Sequence

import arviz as az
import biom
from cmdstanpy import CmdStanModel
import pandas as pd
from patsy import dmatrix

from .inference import fit_to_inference


class BaseModel(ABC):
    """Base BIRDMAn model.

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param model_path: Filepath to Stan model
    :type model_path: str
    """
    def __init__(
        self,
        table: biom.table.Table,
        model_path: str,
    ):
        self.sample_names = table.ids(axis="sample")
        self.model_path = model_path
        self.sm = None
        self.fit = None

        self.dat = {
            "D": table.shape[0],  # number of features
            "N": table.shape[1],  # number of samples
        }

        self.specified = False

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

    def compile_model(self):
        """Compile Stan model."""
        self.sm = CmdStanModel(stan_file=self.model_path)

    def specify_model(
        self,
        params: Sequence[str],
        coords: dict,
        dims: dict,
        include_observed_data: bool = False,
        posterior_predictive: str = None,
        log_likelihood: str = None,
        **kwargs,
    ):
        """Specify coordinates and dimensions of model.

        :param params: Posterior fitted parameters to include
        :type params: Sequence[str]

        :param coords: Mapping of entries in dims to labels
        :type coords: dict

        :param dims: Dimensions of parameters in the model
        :type dims: dict

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

        :param kwargs: Extra keyword arguments to save in specifications dict
        """
        self.params = params
        self.coords = coords
        self.dims = dims
        self.include_observed_data = include_observed_data
        self.posterior_predictive = posterior_predictive
        self.log_likelihood = log_likelihood
        self.specifications = kwargs

        self.specified = True

    def add_parameters(self, param_dict: dict = None):
        """Add parameters from dict to be passed to Stan."""
        if param_dict is None:
            param_dict = dict()
        self.dat.update(param_dict)

    def fit_model(
        self,
        method: str = "vi",
        num_draws: int = 500,
        mcmc_warmup: int = None,
        mcmc_chains: int = 4,
        vi_iter: int = 1000,
        vi_grad_samples: int = 40,
        vi_require_converged: bool = False,
        seed: float = 42,
        mcmc_kwargs: dict = None,
        vi_kwargs: dict = None
    ):
        """Fit BIRDMAn model.

        :param method: Method by which to fit model, either 'mcmc' (default)
            for Markov Chain Monte Carlo or 'vi' for Variational Inference
        :type method: str

        :param num_draws: Number of output draws to sample from the posterior,
            default is 500
        :type num_draws: int

        :param mcmc_warmup: Number of warmup iterations for MCMC sampling,
            default is the same as num_draws
        :type mcmc_warmup: int

        :param mcmc_chains: Number of Markov chains to use for sampling,
            default is 4
        :type mcmc_chains: int

        :param vi_iter: Number of ADVI iterations to use for VI, default is
            1000
        :type vi_iter: int

        :param vi_grad_samples: Number of MC draws for computing the gradient,
            default is 40
        :type vi_grad_samples: int

        :param vi_require_converged: Whether or not to raise an error if Stan
            reports that “The algorithm may not have converged”, default is
            False
        :type vi_require_converged: bool

        :param seed: Random seed to use for sampling, default is 42
        :type seed: int

        :param mcmc_kwargs: kwargs to pass into CmdStanModel.sample

        :param vi_kwargs: kwargs to pass into CmdStanModel.variational
        """
        if method == "mcmc":
            mcmc_kwargs = mcmc_kwargs or dict()
            mcmc_warmup = mcmc_warmup or mcmc_warmup

            self.num_chains = mcmc_chains
            self.num_draws = num_draws

            self.fit = self.sm.sample(
                chains=mcmc_chains,
                parallel_chains=mcmc_chains,
                data=self.dat,
                iter_warmup=mcmc_warmup,
                iter_sampling=num_draws,
                seed=seed,
                **mcmc_kwargs
            )
        elif method == "vi":
            vi_kwargs = vi_kwargs or dict()

            self.num_chains = 1
            self.num_draws = num_draws

            self.fit = self.sm.variational(
                data=self.dat,
                iter=vi_iter,
                output_samples=num_draws,
                grad_samples=vi_grad_samples,
                require_converged=vi_require_converged,
                seed=seed,
                **vi_kwargs
            )
        else:
            raise ValueError("method must be either 'mcmc' or 'vi'")

    @abstractmethod
    def to_inference(self):
        """Convert fitted model to az.InferenceData."""

    def _check_fit_for_inf(self):
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        # if already Inference, just return
        if isinstance(self.fit, az.InferenceData):
            return self.fit

        if not self.specified:
            raise ValueError("Model has not been specified!")


class TableModel(BaseModel):
    """Fit a model on the entire table at once."""
    def __init__(self, table: biom.Table, **kwargs):
        super().__init__(table=table, **kwargs)
        self.feature_names = table.ids(axis="observation")
        self.add_parameters(
            {"y": table.matrix_data.todense().T.astype(int)}
        )

    def to_inference(self) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        self._check_fit_for_inf()

        inference = fit_to_inference(
            fit=self.fit,
            chains=self.num_chains,
            draws=self.num_draws,
            params=self.params,
            coords=self.coords,
            dims=self.dims,
            posterior_predictive=self.posterior_predictive,
            log_likelihood=self.log_likelihood,
            **self.specifications
        )

        if self.include_observed_data:
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
    """Fit a model for a single feature."""
    def __init__(self, table: biom.Table, feature_id: str, **kwargs):
        super().__init__(table=table, **kwargs)
        self.feature_id = feature_id
        values = table.data(
            id=feature_id,
            axis="observation",
            dense=True
        ).astype(int)
        self.add_parameters({"y": values})

    def to_inference(self) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        self._check_fit_for_inf()

        inference = fit_to_inference(
            fit=self.fit,
            chains=self.num_chains,
            draws=self.num_draws,
            params=self.params,
            coords=self.coords,
            dims=self.dims,
            posterior_predictive=self.posterior_predictive,
            log_likelihood=self.log_likelihood,
            **self.specifications
        )

        if self.include_observed_data:
            obs = az.from_dict(
                observed_data={"observed": self.dat["y"]},
                coords={"tbl_sample": self.sample_names},
                dims={"observed": ["tbl_sample"]}
            )
            inference = az.concat(inference, obs)
        return inference


class ModelIterator:
    """Iterate through features in a table.

    This class is intended for those looking to parallelize model fitting
    across individual features rather than across Markov chains.

    :param table: Feature table (features x samples)
    :type table: biom.table.Table

    :param model: BIRDMAn model for each individual feature
    :type model: birdman.model_base.SingleFeatureModel

    :param num_chunks: Number of chunks to split table features. By default
        does not do any chunking.
    :type num_chunks: int

    :param kwargs: Keyword arguments to pass to each feature model
    """
    def __init__(
        self,
        table: biom.Table,
        model: SingleFeatureModel,
        num_chunks: int = None,
        **kwargs
    ):
        self.feature_names = list(table.ids(axis="observation"))
        self.size = table.shape[0]
        self.model_type = model
        self.num_chunks = num_chunks
        models = [model(table, fid, **kwargs) for fid in self.feature_names]

        if num_chunks is None:
            self.chunks = list(zip(self.feature_names, models))
        else:
            chunk_size = ceil(self.size / num_chunks)
            self.chunks = []
            for i in range(0, self.size, chunk_size):
                chunk_feature_names = self.feature_names[i: i+chunk_size]
                chunk_models = models[i: i+chunk_size]

                chunk = [
                    (fid, _model) for fid, _model
                    in zip(chunk_feature_names, chunk_models)
                ]
                self.chunks.append(chunk)

    def __iter__(self):
        return (chunk for chunk in self.chunks)

    def __getitem__(self, chunk_idx: int):
        return self.chunks[chunk_idx]

    def __len__(self):
        return self.num_chunks
