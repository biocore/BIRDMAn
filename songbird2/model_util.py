import pandas as pd

from .model import Model
from .util import alr_to_clr


def collapse_param(model: Model, param: str):
    """Compute mean and stdev for parameter from posterior samples."""
    dfs = []
    res = model.fit.extract(permuted=True)
    colnames = model.dat["dmat"].columns

    # TODO: figure out how to vectorize this
    for i, colname in enumerate(colnames):
        x_clr = alr_to_clr(res[param][:, i, :])
        mean = pd.DataFrame(x_clr.mean(axis=0))
        std = pd.DataFrame(x_clr.std(axis=0))
        df = pd.concat([mean, std], axis=1)
        df.columns = [f"{colname}_{x}" for x in ["mean", "std"]]
        dfs.append(df)
    param_df = pd.concat(dfs, axis=1)
    return param_df
