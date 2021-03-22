from typing import List

import arviz as az
import pandas as pd
import xarray as xr


def ess(
    inference_object: az.InferenceData,
    params: List[str] = None,
    **kwargs
) -> xr.Dataset:
    """Estimate effective sample size for parameters.

    Wrapper function for ``az.ess``.

    See https://arviz-devs.github.io/arviz/api/generated/arviz.ess.html
    for details.

    See `Stan docs <https://mc-stan.org/docs/2_26/reference-manual/\
        effective-sample-size-section.html>`_ for more information on ESS.

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :param params: Variables to include, defaults to all
    :type params: List[str]

    :param kwargs: Keyword arguments to pass to ``az.ess``

    :returns: Estimated effective sample sizes
    :rtype: xr.Dataset
    """
    return az.ess(inference_object, var_names=params, **kwargs)


def rhat(
    inference_object: az.InferenceData,
    params: List[str] = None,
    **kwargs
) -> xr.Dataset:
    """Estimate Gelman-Rubin statistic of chain convergence.

    Wrapper function for ``az.rhat``.

    See https://arviz-devs.github.io/arviz/api/generated/arviz.rhat.html
    for details.

    Rhat values should be very close to 1.0.

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :param params: Variables to include, defaults to all
    :type params: List[str]

    :param kwargs: Keyword arguments to pass to ``az.rhat``

    :returns: Estimated Rhat values
    :rtype: xr.Dataset
    """
    return az.rhat(inference_object, var_names=params, **kwargs)


def loo(inference_object: az.InferenceData, **kwargs) -> az.ELPDData:
    """Compute Pareto-smoothed importance sampling LOO CV.

    Wrapper function for ``az.loo``.

    See https://arviz-devs.github.io/arviz/api/generated/arviz.loo.html
    for details.

    .. note::

        This function requires that the inference object has a
        ``log_likelihood`` group.

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :param kwargs: Keyword arguments to pass to ``az.loo``

    :returns: Estimated log pointwise predictive density
    :rtype: az.ELPDData
    """
    return az.loo(inference_object, **kwargs)


def r2_score(inference_object: az.InferenceData) -> pd.Series:
    """Compute Bayesian :math:`R^2`.

    Wrapper function for ``az.r2_score``.

    .. note::

        This function requires that the inference object has a
        ``posterior_predictive`` group.

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :returns: Bayesian :math:`R^2` & standard deviation
    :rtype: pd.Series
    """
    if "observed_data" not in inference_object.groups():
        raise ValueError("Inference data is missing observed data!")

    y_true = inference_object.observed_data["observed"]
    y_true = y_true.stack(entry=["tbl_sample", "feature"]).data

    pp = inference_object.posterior_predictive
    # Assume only one data variable
    pp_name = list(pp.data_vars)[0]
    y_pred = pp[pp_name].stack(mcmc_sample=["chain", "draw"])
    y_pred = y_pred.stack(entry=["tbl_sample", "feature"]).data
    return az.r2_score(y_true, y_pred)
