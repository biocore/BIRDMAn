from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


def plot_differentials(param_df, covariate):
    """Plot regression parameters.

    Parameters:
    -----------
    param_df : pd.DataFrame
        DataFrame with mean and std for all regression parameters
    covariate : str
        name of covariate to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    d = param_df.shape[0]
    mean = param_df[f"{covariate}_mean"]
    mean = mean.sort_values()
    std = param_df[f"{covariate}_std"]
    std = std.loc[mean.index]

    ax.errorbar(
        np.arange(d),
        mean,
        yerr=std,
        zorder=1,
    )
    ax.scatter(
        np.arange(d),
        mean,
        c="black",
        zorder=2,
    )
    ax.axhline(y=0, linestyle="--", color="black")
    ax.set_ylabel(f"log fold change {covariate}")
    ax.set_xticks([])
    ax.set_xlabel("Features")

    return ax


def plot_ppc(predicted, actual):
    """Plot predictive posterior check.

    Parameters:
    -----------
    predicted : np.ndarray
        predicted values from Stan for each iteration
    actual : np.ndarray
        truth count values
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    n, d = predicted.shape

    predicted_ci = np.quantile(predicted, [0.025, 0.975], axis=0)
    predicted_median = np.median(predicted, axis=0)

    obs_in_ci = (actual >= predicted_ci[0, :]) & (actual <= predicted_ci[1, :])
    pct_in_ci = sum(obs_in_ci)/d*100

    ax.plot(np.arange(d), actual, zorder=2, lw=2, color="black")
    y1, y2 = ax.get_ylim()

    ax.scatter(
        np.arange(d),
        predicted_median,
        color="darkgray",
    )
    for i in range(d):
        ax.plot(
            [i, i],
            predicted_ci[:, i],
            color="lightgray",
            zorder=0,
        )
        pass

    offset = int(d*0.01)
    ax.set_xlim([-offset, d+offset])
    ax.set_ylim([-y2*0.01, y2])

    ax.set_xlabel("Microbe-Sample")
    ax.set_ylabel("Count")

    actual_patch = Line2D([0], [0], color="black", lw=2, label="Actual Count")
    median_patch = Line2D([0], [0], color="darkgray", marker="o", lw=0,
                          markerfacecolor="darkgray",
                          label="Simulated Median")
    interval_patch = Line2D([0], [0], color="lightgray",
                            label="95% Credible Interval")
    handles = [interval_patch, median_patch, actual_patch]
    ax.legend(
        handles=handles,
        loc="upper left",
        edgecolor="black",
        framealpha=1,
    )

    print(f"{round(pct_in_ci, 2)}% of Observations in 95% Credible Interval")
    return ax
