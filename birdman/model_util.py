from typing import Sequence

# import arviz as az
from cmdstanpy import CmdStanMCMC
import numpy as np
import pandas as pd
import xarray as xr

from .util import convert_beta_coordinates


def single_fit_to_xarray(
    fit: CmdStanMCMC,
    params: Sequence,
    feature_names: Sequence,
    covariate_names: Sequence
) -> xr.Dataset:
    """Convert fitted Stan model into xarray Dataset.

    .. note:: Matrix parameters are assumed to be betas and will be
        converted to CLR coordinates.

    :param fit: Fitted model
    :type params: CmdStanMCMC

    :param params: Parameters to include in xarray Dataset
    :type params: Sequence[str]

    :param feature_names: Names of features in feature table
    :type feature_names: Sequence[str]

    :param covariate_names: Names of covariates in design matrix
    :type covariate_names: Sequence[str]

    :returns: xarray Dataset of chosen parameter draws
    :rtype: xr.Dataset

    """
    data_vars = dict()
    for param in params:
        param_draws = fit.stan_variable(param)
        if param_draws.ndim == 3:  # matrix parameter
            param_draws = convert_beta_coordinates(param_draws)

            # Split parameters into individual chains
            # Should be sequential i.e. 0-99 is chain 1, 100-199 is chain 2
            # Becomes (chains x cov x features x draws)
            # TODO: Can probably done smarter with np.reshape
            param_draws = np.array(np.split(param_draws, fit.chains,
                                            axis=2))
            param_coords = ["chain", "covariate", "feature", "draw"]
        elif param_draws.ndim == 2:  # vector parameter
            param_draws = np.array(np.split(param_draws, fit.chains,
                                            axis=0))
            param_coords = ["chain", "draw", "feature"]
        else:
            raise ValueError("Incompatible dimensionality!")
        data_vars[param] = (param_coords, param_draws)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            covariate=covariate_names,
            feature=feature_names,
            draw=np.arange(fit.num_draws_sampling),
            chain=np.arange(fit.chains)
        )
    )
    ds = ds.transpose("covariate", "feature", "chain", "draw")
    return ds


def multiple_fits_to_xarray(
    fits: Sequence[CmdStanMCMC],
    params: Sequence,
    feature_names: Sequence,
    covariate_names: Sequence
) -> xr.Dataset:
    """Save fitted parameters to xarray DataSet for multiple fits.

    :param fits: Fitted models for each feature
    :type params: Sequence[CmdStanMCMC]

    :param params: Parameters to include in xarray Dataset
    :type params: Sequence[str]

    :param feature_names: Names of features in feature table
    :type feature_names: Sequence[str]

    :param covariate_names: Names of covariates in design matrix
    :type covariate_names: Sequence[str]

    :returns: xarray Dataset of chosen parameter draws
    :rtype: xr.Dataset
    """
    assert len(feature_names) == len(fits)

    _fit = fits[0]
    draw_range = np.arange(_fit.num_draws_sampling)
    chain_range = np.arange(_fit.chains)
    param_da_list = []
    # Outer for loop creates DataArray for each parameter
    for param in params:
        all_feat_param_da_list = []
        # Inner for loop creates list of DataArrays for each feature
        for feat, fit in zip(feature_names, fits):
            param_draws = fit.stan_variable(param)  # draw x cov

            param_draws = np.array(np.split(param_draws, fit.chains,
                                            axis=0))
            if param_draws.ndim == 3:  # matrix parameter (chain x draw x cov)
                # not sure if we have to CLR...
                dims = ["chain", "draw", "covariate"]
                coords = [chain_range, draw_range, covariate_names]
            elif param_draws.ndim == 2:  # vector parameter (chain x draw)
                dims = ["chain", "draw"]
                coords = [chain_range, draw_range]
            else:
                raise ValueError("Incompatible dimensionality!")
            feat_param_da = xr.DataArray(  # single feat-param da
                param_draws,
                coords=coords,
                dims=dims,
                name=param,
            )
            all_feat_param_da_list.append(feat_param_da)

        # Concatenates all features for a given parameter to a DataArray
        all_feat_param_da = xr.concat(
            all_feat_param_da_list,
            pd.Index(feature_names, name="feature"),
        )
        param_da_list.append(all_feat_param_da)

    # Merges individual DataArrays for each parameter
    ds = xr.merge(param_da_list)
    ds = ds.transpose("covariate", "feature", "chain", "draw")
    return ds
