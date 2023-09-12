import re
from typing import Sequence

import xarray as xr


def _drop_data(dataset: xr.Dataset, vars_to_drop: Sequence[str]) -> xr.Dataset:
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
