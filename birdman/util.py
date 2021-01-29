import numpy as np


# TODO: Make sure 1-D array works
def alr_to_clr(x: np.ndarray) -> np.ndarray:
    """Convert ALR coordinates to centered CLR coordinates.

    Parameters:
    -----------
    x: np.ndarray
        matrix of ALR coordinates (feature x draw) OR (1 x draw)

    Returns:
    --------
    np.ndarray
        centered CLR coordinates
    """
    if x.ndim == 2:  # matrix of parameter draws
        num_draws = x.shape[1]
    elif x.ndim == 1:  # vector of parameter draws
        num_draws = 1
    else:
        raise ValueError("ALR coordinates must be matrix or vector!")
    z = np.zeros((1, num_draws))
    x_clr = np.vstack((z, x))
    x_clr = x_clr - x_clr.mean(axis=0).reshape(1, -1)
    return x_clr


def convert_beta_coordinates(beta: np.ndarray) -> np.ndarray:
    """Convert feature-covariate coefficients from ALR to CLR.

    Parameters:
    -----------
    beta: np.ndarray
        beta ALR coefficients (p covariates x d features x n draws)

    Returns:
    --------
    beta_clr: np.ndarray
        beta CLR coefficients (p covariates x (d+1) features x n draws)
    """
    num_covariates, num_features, num_draws = beta.shape
    beta_clr = np.zeros((num_covariates, num_features+1, num_draws))
    for i in range(num_covariates):
        beta_clr[i, :, :] = alr_to_clr(beta[i, :, :])
    return beta_clr
