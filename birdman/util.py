from dask.distributed import Client
import numpy as np


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
    if x.ndim == 2:  # matrix of parameters
        num_rows = x.shape[0]
    elif x.ndim == 1:  # vector of parameters
        num_rows = 1
    else:
        raise ValueError("ALR coordinates must be matrix or vector!")
    z = np.zeros((num_rows, 1))
    x_clr = np.hstack((z, x))
    x_clr = x_clr - x_clr.mean(axis=1).reshape(-1, 1)
    return x_clr


def setup_dask_client():
    """Set up dask client & monitoring."""
    client = Client(n_workers=4)
