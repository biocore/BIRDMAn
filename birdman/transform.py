from typing import Sequence

import numpy as np
import xarray as xr

from birdman.util import convert_beta_coordinates


def inference_alr_to_clr(
    posterior: xr.Dataset,
    alr_params: list,
    dim_replacement: dict,
    new_labels: Sequence
) -> xr.Dataset:
    """Convert posterior draws from ALR to CLR.

    :param posterior: Posterior draws for fitted parameters
    :type posterior: xr.Dataset

    :param alr_params: List of parameters to convert from ALR to CLR
    :type alr_params: list

    :param dim_replacement: Dictionary of updated posterior dimension names
        e.g. {"feature_alr": "feature"}
    :type dim_replacement: dict

    :param new_labels: Coordinates to assign to CLR posterior draws
    :type new_labels: Sequence
    """
    new_posterior = posterior.copy()
    for param in alr_params:
        param_da = posterior[param]
        all_chain_alr_coords = param_da
        all_chain_clr_coords = []

        for i, chain_alr_coords in all_chain_alr_coords.groupby("chain"):
            chain_clr_coords = convert_beta_coordinates(chain_alr_coords)
            all_chain_clr_coords.append(chain_clr_coords)

        all_chain_clr_coords = np.array(all_chain_clr_coords)

        new_dims = [
            dim_replacement[x]
            if x in dim_replacement else x
            for x in param_da.dims
        ]
        # Replace coords with updated labels

        new_coords = dict()
        for dim in param_da.dims:
            if dim in dim_replacement:
                new_name = dim_replacement[dim]
                new_coords[new_name] = new_labels
            else:
                new_coords[dim] = param_da.coords.get(dim).data

        new_param_da = xr.DataArray(
            all_chain_clr_coords,
            dims=new_dims,
            coords=new_coords
        )
        new_posterior[param] = new_param_da

    new_posterior = new_posterior.drop_vars(dim_replacement.keys())
    return new_posterior
