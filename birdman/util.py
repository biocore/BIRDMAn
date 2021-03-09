import numpy as np


def alr_to_clr(x: np.ndarray) -> np.ndarray:
    """Convert ALR coordinates to centered CLR coordinates.

    :param x: Matrix of ALR coordinates (features x draws)
    :type x: np.ndarray

    :returns: Matrix of centered CLR coordinates
    :rtype: np.ndarray
    """
    num_draws = x.shape[1]
    z = np.zeros((1, num_draws))
    x_clr = np.vstack((z, x))
    x_clr = x_clr - x_clr.mean(axis=0).reshape(1, -1)
    return x_clr


def clr_to_alr(x: np.ndarray) -> np.ndarray:
    """Convert CLR coordinates to ALR coordinates.

    :param x: Matrix of centered CLR coordinates (features x draws)
    :type x: np.ndarray

    :returns: Matrix of ALR coordinates
    :rtype: np.ndarray
    """
    ref = x[0, :]  # first feature as reference
    return (x - ref)[1:, :]


def convert_beta_coordinates(beta: np.ndarray) -> np.ndarray:
    """Convert feature-covariate coefficients from ALR to CLR.

    :param beta: Matrix of beta ALR coordinates (n draws x p covariates x
        d features)
    :type beta: np.ndarray

    :returns: Matrix of beta CLR coordinates (n draws x p covariates x d+1
        features)
    :rtype: np.ndarray
    """
    num_draws, num_covariates, num_features = beta.shape
    beta_clr = np.zeros((num_draws, num_covariates, num_features+1))
    for i in range(num_covariates):  # TODO: vectorize
        beta_slice = beta[:, i, :].T  # features x draws
        beta_clr[:, i, :] = alr_to_clr(beta_slice).T
    return beta_clr
