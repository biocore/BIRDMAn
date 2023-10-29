from functools import partial
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
    """Convert a fitted model to an arviz InferenceData object.

    :param fit: Fitted CmdStan model
    :type fit: Either CmdStanMCMC or CmdStanVB

    :param chains: Number of chains
    :type chains: int

    :param draws: Number of draws
    :type draws: int

    :param params: Parameters to include in inference
    :type params: Sequence[str]

    :param coords: Coordinates for InferenceData
    :type coords: dict

    :param dims: Dimensions for InferenceData
    :type dims: dict

    :param posterior_predictive: Name of posterior predictive var in model
    :type posterior_predictive: str

    :param log_likelihood: Name of log likelihood var in model
    :type log_likelihood: str

    :returns: Model converted to InferenceData
    :rtype: az.InferenceData
    """
    if log_likelihood is not None and log_likelihood not in dims:
        raise KeyError("Must include dimensions for log-likelihood!")
    if posterior_predictive is not None and posterior_predictive not in dims:
        raise KeyError("Must include dimensions for posterior predictive!")

    # Required because as of writing, CmdStanVB.stan_variable defaults to
    # returning the mean rather than the sample
    if isinstance(fit, CmdStanVB):
        stan_var_fn = partial(fit.stan_variable, mean=False)
    else:
        stan_var_fn = fit.stan_variable

    das = dict()

    for param in params:
        data = stan_var_fn(param)

        _dims = dims[param]
        _coords = {k: coords[k] for k in _dims}

        das[param] = stan_var_to_da(data, _coords, _dims, chains, draws)

    if log_likelihood:
        data = stan_var_fn(log_likelihood)

        _dims = dims[log_likelihood]
        _coords = {k: coords[k] for k in _dims}

        ll_da = stan_var_to_da(data, _coords, _dims, chains, draws)
        ll_ds = xr.Dataset({log_likelihood: ll_da})
    else:
        ll_ds = None

    if posterior_predictive:
        data = stan_var_fn(posterior_predictive)

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
    if "log_likelihood" in inf_list[0].groups():
        group_list.append([x.log_likelihood for x in inf_list])
    if "posterior_predictive" in inf_list[0].groups():
        group_list.append([x.posterior_predictive for x in inf_list])

    po_ds = xr.concat(group_list[0], concatenation_name)
    group_dict = {"posterior": po_ds}

    if "log_likelihood" in inf_list[0].groups():
        ll_ds = xr.concat(group_list[1], concatenation_name)
        group_dict["log_likelihood"] = ll_ds
    if "posterior_predictive" in inf_list[0].groups():
        pp_ds = xr.concat(group_list[2], concatenation_name)
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
    """Convert results of stan_var to DataArray.

    :params data: Result of stan_var
    :type data: np.ndarray

    :params coords: Coordinates of variables
    :type coords: dict

    :params dims: Dimensions of variables
    :type dims: dict

    :params chains: Number of chains
    :type chains: int

    :params draws: Number of draws
    :type draws: int

    :returns: DataArray representation of stan variables
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
