from typing import List, Sequence, Union

import arviz as az
from cmdstanpy import CmdStanMCMC, CmdStanVB
import numpy as np
import xarray as xr


def fit_to_inference(
    fit: Union[CmdStanMCMC, CmdStanVB],
    chains: int,
    draws: int,
    params: Sequence[str],
    coords: dict,
    dims: dict,
    posterior_predictive: str = None,
    log_likelihood: str = None,
) -> az.InferenceData:
    """Convert fitted Stan object to ArviZ InferenceData.

    :param fit: Fitted model, either from MCMC or VI
    :type fit: CmdStanMCMC or CmdStanVB

    :param chains: Number of chains used for sampling
    :type chains: int

    :param draws: Number of posterior draws
    :type draws: int

    :param params: Posterior fitted parameters to include
    :type params: Sequence[str]

    :param coords: Mapping of entries in dims to labels
    :type coords: dict

    :param dims: Dimensions of parameters in the model
    :type dims: dict

    :param posterior_predictive: Name of posterior predictive values from
        Stan model to include in ``arviz`` InferenceData object
    :type posterior_predictive: str, optional

    :param log_likelihood: Name of log likelihood values from Stan model
        to include in ``arviz`` InferenceData object
    :type log_likelihood: str, optional

    :returns: Fit in InferenceData format
    :rtype: az.InferenceData
    """
    if log_likelihood is not None and log_likelihood not in dims:
        raise KeyError("Must include dimensions for log-likelihood!")
    if posterior_predictive is not None and posterior_predictive not in dims:
        raise KeyError("Must include dimensions for posterior predictive!")

    das = dict()

    for param in params:
        data = fit.stan_variable(param)

        _dims = dims[param]
        _coords = {k: coords[k] for k in _dims}

        das[param] = stan_var_to_da(data, _coords, _dims, chains, draws)

    if log_likelihood:
        data = fit.stan_variable(log_likelihood)

        _dims = dims[log_likelihood]
        _coords = {k: coords[k] for k in _dims}

        ll_da = stan_var_to_da(data, _coords, _dims, chains, draws)
        ll_ds = xr.Dataset({log_likelihood: ll_da})
    else:
        ll_ds = None

    if posterior_predictive:
        data = fit.stan_variable(posterior_predictive)

        _dims = dims[posterior_predictive]
        _coords = {k: coords[k] for k in _dims}

        pp_da = stan_var_to_da(data, _coords, _dims, chains, draws)
        pp_ds = xr.Dataset({posterior_predictive: pp_da})
    else:
        pp_ds = None

    inf = az.InferenceData(
        posterior=xr.Dataset(das),
        log_likelihood=ll_ds,
        posterior_predictive=pp_ds
    )

    return inf


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


def stan_var_to_da(
    data: np.ndarray,
    coords: dict,
    dims: dict,
    chains: int,
    draws: int
) -> xr.DataArray:
    """Convert Stan variable draws to xr.DataArray.

    :param data: Output of stan_variable method from VB or MCMC
    :type data: np.ndarray

    :param coords: Mapping of entries in dims to labels
    :type coords: dict

    :param dims: Dimensions of parameters in the model
    :type dims: dict

    :param chains: Number of chains included in data
    :type chains: int

    :param draws: Number of posterio draws
    :type draws: int

    :returns: xarray DataArray of posterior draws split by chain & draw
    :rtype: xr.DataArray
    """
    data = np.stack(np.split(data, chains))

    coords["draw"] = np.arange(draws)
    coords["chain"] = np.arange(chains)
    dims = ["chain", "draw"] + dims

    da = xr.DataArray(
        data,
        coords=coords,
        dims=dims,
    )

    return da
