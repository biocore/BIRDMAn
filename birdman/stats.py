from typing import Tuple

import arviz as az
import numpy as np
from scipy.stats import f as f_distrib

from .util import clr_to_alr


def hotelling_ttest(
    inference_object: az.InferenceData,
    coord: dict,
    parameter: str = "beta",
) -> Tuple[np.float64, np.float64]:
    """Test if covariate-draws centered around zero.

    Uses `Hotelling's T-squared test\
        <https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution>`_\
        to determine whether a given covariate coefficient is centered around\
        zero (null hypothesis: centered around 0).

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :param coord: Coordinates to test
    :type coord: dict

    :param parameter: Name of parameter to test, defaults to 'beta'
    :type parameter: str, optional

    :returns: :math:`t^2` & p-value
    :rtype: Tuple(float, float)
    """
    data = inference_object.posterior[parameter].sel(coord)
    data = data.stack(sample=("chain", "draw")).data
    return _hotelling(data)


def _hotelling(x: np.ndarray) -> Tuple[np.float64, np.float64]:
    """Calculate Hotelling test statistic and p-value.

    :param x: Centered CLR data matrix (features x draws)
    :type x: np.ndarray

    :returns: :math:`t^2` & p-value
    :rtype: Tuple(float, float)
    """
    x_alr = clr_to_alr(x)  # Can't use CLR b/c covariance matrix is singular
    num_feats, num_draws = x_alr.shape

    if num_feats > num_draws:
        msg = "Number of samples must be larger than number of features!"
        raise ValueError(msg)

    mu = x_alr.mean(axis=1)
    cov = np.cov(x_alr)
    inv_cov = np.linalg.pinv(cov)
    t2 = mu @ inv_cov @ mu.T
    stat = (t2 * (num_draws - num_feats) / ((num_draws - 1) * num_feats))
    npval = np.squeeze(f_distrib.cdf(stat, num_feats, num_draws - num_feats))
    pval = 1 - npval
    return t2, pval
