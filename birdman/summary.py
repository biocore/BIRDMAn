from typing import Union

import pandas as pd
import xarray as xr


def summarize_posterior(
    posterior: xr.Dataset,
    data_var: str,
    coords: dict = None,
    estimator: str = "mean"
) -> Union[pd.DataFrame, pd.Series]:
    """Summarize posterior distribution to pandas.

    :param posterior: Posterior distribution of estimated parameters
    :type posterior: xr.Dataset

    :param data_var: Data variable to summarize
    :type data_var: str

    :param coords: Mapping of entries in dims to labels
    :type coords: dict

    :param estimator: How to summarize posterior, one of 'mean', 'median', or
        'std'
    :type estimator: str

    :returns: Summarized posterior according to estimator
    :rtype: pd.DataFrame or pd.Series
    """
    coords = dict() or coords
    data = posterior[data_var].sel(coords)

    if estimator == "mean":
        func = data.mean
    elif estimator == "median":
        func = data.median
    elif estimator == "std":
        func = data.std
    else:
        raise ValueError(
            "estimator must be either 'mean', 'median', or 'std'"
        )

    summ = func(["chain", "draw"])
    new_dims = list(summ.dims)
    df = summ.transpose(*new_dims).to_pandas()

    # Assume largest dimension is feature
    if isinstance(df, pd.DataFrame):
        x, y = df.shape
        if y > x:
            df = df.T

    return df
