import pandas as pd

from .model import Model
from .util import alr_to_clr


def collapse_param(model: Model, param: str, convert_alr_to_clr=False):
    """Compute mean and stdev for parameter from posterior samples."""
    dfs = []
    res = model.fit.extract(permuted=True)

    param_data = res[param]

    # TODO: figure out how to vectorize this
    if param_data.ndim == 3:  # matrix parameter
        for i, colname in enumerate(model.colnames):
            if convert_alr_to_clr:  # for beta parameters
                data = alr_to_clr(param_data[:, i, :])
            else:
                data = param_data
            mean = pd.DataFrame(data.mean(axis=0))
            std = pd.DataFrame(data.std(axis=0))
            df = pd.concat([mean, std], axis=1)
            df.columns = [f"{colname}_{x}" for x in ["mean", "std"]]
            dfs.append(df)
        param_df = pd.concat(dfs, axis=1)
    elif param_data.ndim == 2:  # vector parameter
        if convert_alr_to_clr:
            data = alr_to_clr(param_data)
        else:
            data = param_data
        mean = pd.DataFrame(data.mean(axis=0))
        std = pd.DataFrame(data.std(axis=0))
        param_df = pd.concat([mean, std], axis=1)
        param_df.columns = [f"{param}_{x}" for x in ["mean", "std"]]
    else:
        raise ValueError("Parameter must be matrix or vector type!")
    return param_df
