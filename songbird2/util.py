import numpy as np
import pandas as pd


def alr_to_clr(x):
    """Convert ALR coordinates to centered CLR coordinates.

    Parameters:
    -----------
    x: np.ndarray
        matrix of ALR coordinates

    Returns:
    --------
    np.ndarray
        centered CLR coordinates
    """
    num_rows = x.shape[0]
    z = np.zeros((num_rows, 1))
    x_clr = np.hstack((z, x))
    x_clr = x_clr - x_clr.mean(axis=1).reshape(-1, 1)
    return x_clr


def collapse_results(beta, colnames):
    """Compute mean and stdev for parameters from posterior samples."""
    dfs = []

    # TODO: figure out how to vectorize this
    for i, colname in enumerate(colnames):
        x_clr = alr_to_clr(beta[:, i, :])
        mean = pd.DataFrame(x_clr.mean(axis=0))
        std = pd.DataFrame(x_clr.std(axis=0))
        df = pd.concat([mean, std], axis=1)
        df.columns = [f"{colname}_{x}" for x in ["mean", "std"]]
        dfs.append(df)
    beta_df = pd.concat(dfs, axis=1)
    return beta_df
