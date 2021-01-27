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


def extract_fit_diagnostics(fit):
    """Extract Rhat and n_eff from fitted Stan model.

    https://github.com/stan-dev/pystan/issues/396#issuecomment-343018644
    """
    summary_colnames = fit.summary()["summary_colnames"]
    rhat_index = summary_colnames.index("Rhat")
    neff_index = summary_colnames.index("n_eff")
    rhat = fit.summary()["summary"][:, rhat_index]
    neff = fit.summary()["summary"][:, neff_index]
    return rhat, neff
