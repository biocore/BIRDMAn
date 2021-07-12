import re
from typing import List, Sequence

import arviz as az
from cmdstanpy import CmdStanMCMC
import xarray as xr


def full_fit_to_inference(
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

    :param alr_params: Parameters to convert from ALR to CLR
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
    if log_likelihood is not None and log_likelihood not in dims:
        raise KeyError("Must include dimensions for log-likelihood!")
    if posterior_predictive is not None and posterior_predictive not in dims:
        raise KeyError("Must include dimensions for posterior predictive!")

    inference = az.from_cmdstanpy(
        fit,
        coords=coords,
        log_likelihood=log_likelihood,
        posterior_predictive=posterior_predictive,
        dims=dims
    )

    vars_to_drop = set(inference.posterior.data_vars).difference(params)
    inference.posterior = _drop_data(inference.posterior, vars_to_drop)

    return inference


def single_feature_fit_to_inference(
    fit: CmdStanMCMC,
    params: Sequence[str],
    coords: dict,
    dims: dict,
    posterior_predictive: str = None,
    log_likelihood: str = None,
) -> az.InferenceData:
    """Convert single feature fit to InferenceData.

    :param fit: Single feature fit with CmdStanPy
    :type fit: cmdstanpy.CmdStanMCMC

    :param params: Posterior fitted parameters to include
    :type params: Sequence[str]

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
    _coords = coords.copy()
    if "feature" in coords:
        _coords.pop("feature")

    _dims = dims.copy()
    for k, v in _dims.items():
        if "feature" in v:
            v.remove("feature")

    feat_inf = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive,
        log_likelihood=log_likelihood,
        coords=_coords,
        dims=_dims
    )
    vars_to_drop = set(feat_inf.posterior.data_vars).difference(params)
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
