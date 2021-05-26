import re
from typing import List, Sequence, Union

import arviz as az
from cmdstanpy import CmdStanMCMC
import dask
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
) -> az.InferenceData:
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

    if log_likelihood is not None and log_likelihood not in dims:
        raise KeyError("Must include dimensions for log-likelihood!")
    if posterior_predictive is not None and posterior_predictive not in dims:
        raise KeyError("Must include dimensions for posterior predictive!")

    inference = az.from_cmdstanpy(
        fit,
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
        mcmc_coords = {
            "chain": np.arange(fit.chains),
            "draw": np.arange(fit.num_draws_sampling)
        }
        # restrict param DataArray to only required dims/coords
        tmp_coords = {k: coords[k] for k in dims[param]}
        param_da = xr.DataArray(
            all_chain_clr_coords,
            dims=tmp_dims,
            coords={**tmp_coords, **mcmc_coords}
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
    concatenate: bool = True,
    concatenation_name: str = "feature",
    posterior_predictive: str = None,
    log_likelihood: str = None,
) -> Union[az.InferenceData, List[az.InferenceData]]:
    """Save fitted parameters to xarray DataSet for multiple fits.

    :param fits: Fitted models for each feature
    :type params: Sequence[CmdStanMCMC]

    :param params: Posterior fitted parameters to include
    :type params: Sequence[str]

    :param coords: Mapping of entries in dims to labels
    :type coords: dict

    :param dims: Dimensions of parameters in the model
    :type dims: dict

    :param concatenate: Whether to concatenate all fits together, defaults to
        True
    :type concatenate: bool

    :param_concatenation_name: Name to aggregate features when combining
        multiple fits, defaults to 'feature'
    :type concatentation_name: str, optional

    :param posterior_predictive: Name of posterior predictive values from
        Stan model to include in ``arviz`` InferenceData object
    :type posterior_predictive: str, optional

    :param log_likelihood: Name of log likelihood values from Stan model
        to include in ``arviz`` InferenceData object
    :type log_likelihood: str, optional

    :returns: ``arviz`` InferenceData object with selected values
    :rtype: az.InferenceData
    """
    # Remove the concatenation dimension from each individual fit if
    # necessary
    new_dims = {
        k: [dim for dim in v if dim != concatenation_name]
        for k, v in dims.items()
    }

    # Remove the unnecessary posterior dimensions
    all_vars = fits[0].stan_variables().keys()
    vars_to_drop = set(all_vars).difference(params)
    if log_likelihood is not None:
        vars_to_drop.remove(log_likelihood)
    if posterior_predictive is not None:
        vars_to_drop.remove(posterior_predictive)

    inf_list = []
    for fit in fits:
        single_feat_inf = dask.delayed(_single_feature_to_inf)(
            fit=fit,
            coords=coords,
            dims=new_dims,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            vars_to_drop=vars_to_drop
        )
        inf_list.append(single_feat_inf)

    inf_list = dask.compute(*inf_list)
    if not concatenate:  # Return list of individual InferenceData objects
        return inf_list
    else:
        return concatenate_inferences(inf_list, coords, concatenation_name)


def _single_feature_to_inf(
    fit: CmdStanMCMC,
    coords: dict,
    dims: dict,
    vars_to_drop: Sequence[str],
    posterior_predictive: str = None,
    log_likelihood: str = None,
) -> az.InferenceData:
    """Convert single feature fit to InferenceData.

    :param fit: Single feature fit with CmdStanPy
    :type fit: cmdstanpy.CmdStanMCMC

    :param coords: Coordinates to use for annotating Inference dims
    :type coords: dict

    :param dims: Dimensions of parameters in fitted model
    :type dims: dict

    :param posterior_predictive: Name of variable holding PP values
    :type posterior_predictive: str

    :param log_likelihood: Name of variable holding LL values
    :type log_likelihood: str

    :returns: InferenceData object of single feature
    :rtype: az.InferenceData
    """
    feat_inf = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims
    )
    feat_inf.posterior = _drop_data(feat_inf.posterior, vars_to_drop)
    return feat_inf


def concatenate_inferences(
    inf_list: List[az.InferenceData],
    coords: dict,
    concatenation_name: str = "feature"
) -> az.InferenceData:
    """Concatenates multiple single feature fits into one object.

    :param inf_list: List of InferenceData objects for each feature
    :type inf_list: List[az.InferenceData]

    :param coords: Coordinates containing concatenation name labels
    :type coords: dict

    :param concatenation_name: Name of feature dimension used when
        concatenating, defaults to "feature"
    :type concatenation_name: str

    :returns: Combined InferenceData object
    :rtype: az.InferenceData
    """
    group_list = []
    group_list.append([x.posterior for x in inf_list])
    group_list.append([x.sample_stats for x in inf_list])
    if "log_likelihood" in inf_list[0].groups():
        group_list.append([x.log_likelihood for x in inf_list])
    if "posterior_predictive" in inf_list[0].groups():
        group_list.append([x.posterior_predictive for x in inf_list])

    po_ds = xr.concat(group_list[0], concatenation_name)
    ss_ds = xr.concat(group_list[1], concatenation_name)
    group_dict = {"posterior": po_ds, "sample_stats": ss_ds}

    if "log_likelihood" in inf_list[0].groups():
        ll_ds = xr.concat(group_list[2], concatenation_name)
        group_dict["log_likelihood"] = ll_ds
    if "posterior_predictive" in inf_list[0].groups():
        pp_ds = xr.concat(group_list[3], concatenation_name)
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
    vars_to_drop: Sequence[str],
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
