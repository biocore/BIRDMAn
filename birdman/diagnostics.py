from typing import List

import arviz as az
import xarray as xr


def ess(
    inference_object: az.InferenceData,
    params: List[str] = None,
    **kwargs
) -> xr.Dataset:
    """Estimate effective sample size for parameters.

    Wrapper function for ``az.ess``. See `documentation <https://arviz-devs.\
        github.io/arviz/api/generated/arviz.ess.html>`_ for details.

    See `Stan docs <https://mc-stan.org/docs/2_26/reference-manual/\
        effective-sample-size-section.html>`_ for more information.

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :param params: Variables to include, defaults to all
    :type params: List[str]

    :param kwargs: Keyword arguments to pass to ``az.ess``
    """
    return az.ess(inference_object, var_names=params, **kwargs)


def rhat(
    inference_object: az.InferenceData,
    params: List[str] = None,
    **kwargs
) -> xr.Dataset:
    """Estimate Gelman-Rubin statistic of chain convergence.

    Wrapper function for ``az.rhat``. See `documentation <https://arviz-devs.\
        github.io/arviz/api/generated/arviz.rhat.html>`_ for details.

    Rhat values should be very close to 1.0.

    :param inference_object: Inference object with posterior draws
    :type inference_object: az.InferenceData

    :param params: Variables to include, defaults to all
    :type params: List[str]

    :param kwargs: Keyword arguments to pass to ``az.rhat``
    """
    return az.rhat(inference_object, var_names=params, **kwargs)
