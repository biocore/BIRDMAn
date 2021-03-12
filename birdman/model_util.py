import re
from typing import Sequence

import arviz as az
from cmdstanpy import CmdStanMCMC
import numpy as np
import xarray as xr

from .util import convert_beta_coordinates


def single_fit_to_inference(
    fit: CmdStanMCMC,
    params: Sequence[str],
    coords: dict,
    dims: dict,
    alr_params: Sequence[str] = None,
    posterior_predictive: str = None,
    log_likelihood: str = None,
    sample_names: Sequence[str] = None,
) -> xr.Dataset:
    """Convert fitted Stan model into inference object.

    :param fit: Fitted model
    :type params: CmdStanMCMC

    :param params: Posterior fitted parameters to include
    :type params: Sequence[str]

    :param coords: Mapping of entries in dims to labels
    :type coords: dict

    :param dims: Dimensions of parameters in the model
    :type dims: dict

    :param alr_params: Parameters to convert from ALR to CLR (this will
        be ignored if the model has been parallelized across features)
    :type alr_params: Sequence[str], optional

    :param posterior_predictive: Name of posterior predictive values from
        Stan model to include in ``arviz`` InferenceData object
    :type posterior_predictive: str, optional

    :param log_likelihood: Name of log likelihood values from Stan model
        to include in ``arviz`` InferenceData object
    :type log_likelihood: str, optional

    :returns: ``arviz`` InferenceData object with selected values
    :rtype: az.InferenceData
    """
    # remove alr params so initial dim fitting works
    new_dims = {k: v for k, v in dims.items() if k not in alr_params}
    inference = az.from_cmdstanpy(
        posterior=fit,
        coords=coords,
        log_likelihood=log_likelihood,
        posterior_predictive=posterior_predictive,
        dims=new_dims
    )

    vars_to_drop = set(inference.posterior.data_vars).difference(params)
    inference.posterior = _drop_data(inference.posterior, vars_to_drop)

    # Convert each param in ALR coordinates to CLR coordinates
    for param in alr_params:
        # Want to run on each chain independently
        all_chain_clr_coords = []
        all_chain_alr_coords = np.split(fit.stan_variable(param), fit.chains,
                                        axis=0)
        for i, chain_alr_coords in enumerate(all_chain_alr_coords):
            # arviz 0.11.2 seems to flatten for some reason even though
            # the PR was specifically supposed to do the opposite.
            # Not sure what's going on but just going to go through cmdstanpy.
            chain_clr_coords = convert_beta_coordinates(chain_alr_coords)
            all_chain_clr_coords.append(chain_clr_coords)
        all_chain_clr_coords = np.array(all_chain_clr_coords)

        tmp_dims = ["chain", "draw"] + dims[param]
        chain_coords = {"chain": np.arange(fit.chains)}
        draw_coords = {"draw": np.arange(fit.num_draws_sampling)}
        param_da = xr.DataArray(
            all_chain_clr_coords,
            dims=tmp_dims,
            coords={**coords, **chain_coords, **draw_coords}
        )
        inference.posterior[param] = param_da

        # TODO: Clean this up
        all_dims = list(inference.posterior.dims)
        dims_to_drop = []
        for dim in all_dims:
            if re.match(f"{param}_dim_\\d", dim):
                dims_to_drop.append(dim)
        inference.posterior = inference.posterior.drop_dims(dims_to_drop)
    return inference


def multiple_fits_to_inference(
    fits: Sequence[CmdStanMCMC],
    params: Sequence[str],
    coords: dict,
    dims: dict,
    concatenation_name: str,
    posterior_predictive: str = None,
    log_likelihood: str = None,
    sample_names: Sequence[str] = None
) -> az.InferenceData:
    """Save fitted parameters to xarray DataSet for multiple fits.

    :param fits: Fitted models for each feature
    :type params: Sequence[CmdStanMCMC]

    :param params: Posterior fitted parameters to include
    :type params: Sequence[str]

    :param coords: Mapping of entries in dims to labels
    :type coords: dict

    :param dims: Dimensions of parameters in the model
    :type dims: dict

    :param_concatenation_name: Name to aggregate features when combining
        multiple fits, defaults to 'feature'
    :type concatentation_name: str, optional

    :param posterior_predictive: Name of posterior predictive values from
        Stan model to include in ``arviz`` InferenceData object
    :type posterior_predictive: str, optional

    :param log_likelihood: Name of log likelihood values from Stan model
        to include in ``arviz`` InferenceData object
    :type log_likelihood: str, optional

    :param sample_names: Names of table samples
    :type sample_names: Sequence[str], optional

    :returns: ``arviz`` InferenceData object with selected values
    :rtype: az.InferenceData
    """
    # Remove the concatenation dimension from each individual fit
    new_dims = {
        k: [dim for dim in v if dim != concatenation_name]
        for k, v in dims.items()
    }

    # If dims are unchanged it means the concatenation_name was not found
    if new_dims == dims:
        raise ValueError("concatenation_name must match dimensions in dims")

    po_list = []  # posterior
    ss_list = []  # sample stats
    pp_list = []  # posterior predictive
    ll_list = []  # log likelihood
    for fit in fits:
        ds = az.from_cmdstanpy(
            posterior=fit,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            coords=coords,
            dims=new_dims
        )
        vars_to_drop = set(ds.posterior.data_vars).difference(params)
        ds.posterior = _drop_data(ds.posterior, vars_to_drop)

        po_list.append(ds.posterior)
        ss_list.append(ds.sample_stats)
        if log_likelihood is not None:
            ll_list.append(ds.log_likelihood)
        if posterior_predictive is not None:
            pp_list.append(ds.posterior_predictive)

    po_ds = xr.concat(po_list, concatenation_name)
    ss_ds = xr.concat(ss_list, concatenation_name)
    group_dict = {"posterior": po_ds, "sample_stats": ss_ds}

    # Flatten table values for use in arviz built-in stats
    if log_likelihood is not None:
        ll_ds = xr.concat(ll_list, concatenation_name)
        ll_data = ll_ds[log_likelihood].values
        ll_ds = ll_ds.drop_vars([log_likelihood])  # drop to save mem
        ll_data = ll_data.reshape(fit.chains, fit.num_draws_sampling, -1)
        ll_ds = ll_ds.assign_coords(table_entry=np.arange(ll_data.shape[-1]))
        ll_ds[log_likelihood] = (("chain", "draw", "table_entry"),
                                 ll_data)
        ll_dim_name = f"{log_likelihood}_dim_0"
        ll_ds = ll_ds.drop_dims([ll_dim_name])
        group_dict["log_likelihood"] = ll_ds
    if posterior_predictive is not None:
        pp_ds = xr.concat(pp_list, concatenation_name)
        pp_data = pp_ds[posterior_predictive].values
        pp_ds = pp_ds.drop_vars([posterior_predictive])  # drop to save mem
        pp_data = pp_data.reshape(fit.chains, fit.num_draws_sampling, -1)
        pp_ds = pp_ds.assign_coords(table_entry=np.arange(pp_data.shape[-1]))
        pp_ds[posterior_predictive] = (("chain", "draw", "table_entry"),
                                       pp_data)
        pp_dim_name = f"{posterior_predictive}_dim_0"
        pp_ds = pp_ds.drop_dims([pp_dim_name])
        group_dict["posterior_predictive"] = pp_ds

    all_group_inferences = []
    for group in group_dict:
        # Set concatenation dim coords
        group_ds = group_dict[group].assign_coords(
            {concatenation_name: coords[concatenation_name]}
        )

        group_inf = az.InferenceData(**{group: group_ds})  # hacky
        all_group_inferences.append(group_inf)

    return az.concat(*all_group_inferences)


def _drop_data(
    dataset: xr.Dataset,
    vars_to_drop: Sequence
) -> xr.Dataset:
    """Drop data and associated dimensions from inference group."""
    new_dataset = dataset.drop_vars(vars_to_drop)
    # TODO: Figure out how to do this more cleanly
    dims_to_drop = []
    for var in vars_to_drop:
        for dim in new_dataset.dims:
            if re.match(f"{var}_dim_\\d", dim):
                dims_to_drop.append(dim)
    new_dataset = new_dataset.drop_dims(dims_to_drop)
    return new_dataset
