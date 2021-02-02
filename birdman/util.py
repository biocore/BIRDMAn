import numpy as np


def alr_to_clr(x: np.ndarray) -> np.ndarray:
    """Convert ALR coordinates to centered CLR coordinates.

    Parameters:
    -----------
    x: np.ndarray
        matrix of ALR coordinates (features x draws)

    Returns:
    --------
    np.ndarray
        centered CLR coordinates
    """
    num_draws = x.shape[1]
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
    np.ndarray
        beta CLR coefficients (p covariates x (d+1) features x n draws)
    """
    num_covariates, num_features, num_draws = beta.shape
    beta_clr = np.zeros((num_covariates, num_features+1, num_draws))
    for i in range(num_covariates):  # TODO: vectorize
        beta_clr[i, :, :] = alr_to_clr(beta[i, :, :])
    return beta_clr
