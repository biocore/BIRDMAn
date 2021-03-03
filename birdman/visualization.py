import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def plot_differential_intervals(
    inference_object: az.InferenceData,
    parameter: str,
    coord: dict,
    num_std: float = 1.0
):
    """Plot credible intervals of estimated parameters.

    :param inference_object: Inference object containing posterior draws
    :type inference_object: az.InferenceData

    :param parameter: Name of parameter to plot
    :type parameter: str

    :param coord: Coordinates of parameter to plot
    :type coord: dict

    :param num_std: Number of standard deviations to plot as error bars
    :type num_std: float
    """
    posterior = inference_object.posterior
    param_medians = posterior[parameter].sel(**coord).median(["chain", "draw"])
    param_stds = posterior[parameter].sel(**coord).std(["chain", "draw"])
    sort_indices = param_medians.argsort().data
    param_medians = param_medians.data[sort_indices]
    param_stds = param_stds.data[sort_indices]

    x = np.arange(len(param_medians))

    fig, ax = plt.subplots(1, 1)
    ax.axhline(y=0, color="black", linestyle="--")
    ax.errorbar(x=x, y=param_medians, yerr=param_stds*num_std)
    ax.scatter(x=x, y=param_medians)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Differential")

    return ax
