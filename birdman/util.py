from typing import List

from cmdstanpy.stanfit import CmdStanMCMC
import numpy as np
import pandas as pd
import xarray as xr


def alr_to_clr(x: np.ndarray) -> np.ndarray:
    """Convert ALR coordinates to centered CLR coordinates.

    :param x: Matrix of ALR coordinates (features x draws)
    :type x: np.ndarray

    :returns: Matrix of centered CLR coordinates
    :rtype: np.ndarray
    """
    num_draws = x.shape[1]
    z = np.zeros((1, num_draws))
    x_clr = np.vstack((z, x))
    x_clr = x_clr - x_clr.mean(axis=0).reshape(1, -1)
    return x_clr


def convert_beta_coordinates(beta: np.ndarray) -> np.ndarray:
    """Convert feature-covariate coefficients from ALR to CLR.

    :param beta: Matrix of beta ALR coordinates (n draws x p covariates x
        d features)
    :type beta: np.ndarray

    :returns: Matrix of beta CLR coordinates (p covariates x (d+1) features x
        n draws)
    :rtype: np.ndarray
    """
    # axis moving is an artifact of previous PyStan implementation
    # want dims to be (p covariates x d features x n draws)
    # TODO: make this function work on the original dimensions
    beta = np.moveaxis(beta, [1, 2, 0], [0, 1, 2])
    num_covariates, num_features, num_draws = beta.shape
    beta_clr = np.zeros((num_covariates, num_features+1, num_draws))
    for i in range(num_covariates):  # TODO: vectorize
        beta_clr[i, :, :] = alr_to_clr(beta[i, :, :])
    return beta_clr


# TODO: Combine fit_to_xarray and fits_to_xarray
# TODO: Add flag for which parameters to convert to CLR
# NOTE: May be better to put into Model class instead of util module
def fit_to_xarray(
    fit: CmdStanMCMC,
    params: list,
    feature_names: list,
    covariate_names: list
) -> xr.Dataset:
    """Save fitted parameters to xarray DataSet.

    .. note:: Matrix parameters are assumed to be betas and will be converted
       to CLR coordinates.

    :param fit: Fitted cmdstan object
    :type fit: CmdStanMCMC

    :param params: Parameters to include in xarray Dataset
    :type params: list[str]

    :param feature_names: Names of features in feature table
    :type feature_names: list[str]

    :param covariate_names: Names of covariates in design matrix
    :type covariate_names: list[str]

    :returns: xarray Dataset of chosen parameter draws
    :rtype: xr.Dataset
    """
    # TODO: Figure out how to handle parallelized model
    data_vars = dict()
    for param in params:
        param_draws = fit.stan_variable(param)
        if param_draws.ndim == 3:  # matrix parameter
            param_coords = ["covariate", "feature", "draw"]
            param_draws = convert_beta_coordinates(param_draws)
        elif param_draws.ndim == 2:  # vector parameter
            param_coords = ["draw", "feature"]
        else:
            raise ValueError("Incompatible dimensionality!")
        data_vars[param] = (param_coords, param_draws)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            covariate=covariate_names,
            feature=feature_names,
            draw=np.arange(fit.num_draws_sampling*fit.chains)
        )
    )
    ds = ds.transpose("covariate", "feature", "draw")
    return ds


def fits_to_xarray(
    fits: List[CmdStanMCMC],
    params: list,
    feature_names: list,
    covariate_names: list
) -> xr.Dataset:
    """Save fitted parameters to xarray DataSet for multiple fits.

    :param fits: List of fitted cmdstan object
    :type fits: list[CmdStanMCMC]

    :param params: Parameters to include in xarray Dataset
    :type params: list[str]

    :param feature_names: Names of features in feature table
    :type feature_names: list[str]

    :param covariate_names: Names of covariates in design matrix
    :type covariate_names: list[str]

    :returns: xarray Dataset of chosen parameter draws
    :rtype: xr.Dataset
    """
    assert len(feature_names) == len(fits)

    _fit = fits[0]
    draw_range = np.arange(_fit.num_draws_sampling*_fit.chains)
    param_da_list = []
    for param in params:
        all_feat_param_da_list = []
        for feat, fit in zip(feature_names, fits):
            param_draws = fit.stan_variable(param)  # draw x cov

            if param_draws.ndim == 2:  # matrix parameter
                # not sure if we have to CLR...
                dims = ["draw", "covariate"]
                coords = [draw_range, covariate_names]
            elif param_draws.ndim == 1:  # vector parameter
                dims = ["draw"]
                coords = [draw_range]
            else:
                raise ValueError("Incompatible dimensionality!")
            feat_param_da = xr.DataArray(  # single feat-param da
                param_draws,
                coords=coords,
                dims=dims,
                name=param,
            )
            all_feat_param_da_list.append(feat_param_da)

        all_feat_param_da = xr.concat(
            all_feat_param_da_list,
            pd.Index(feature_names, name="feature"),
        )
        param_da_list.append(all_feat_param_da)

    ds = xr.merge(param_da_list)
    ds = ds.transpose("covariate", "feature", "draw")
    return ds
