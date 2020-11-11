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
    num_rows = x.shape[0]
    z = np.zeros((num_rows, 1))
    x_clr = np.hstack(z, x)
    x_clr = x_clr - x_clr.mean(axis=1)
    return x_clr


def process_beta(beta):
    """Computes mean and std for beta values from posterior draws.
    """
    return
