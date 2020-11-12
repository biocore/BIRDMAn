import numpy as np
import matplotlib.pyplot as plt


def plot_differentials(beta_df, covariate):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    d = beta_df.shape[0]
    mean = beta_df[f"{covariate}_mean"]
    mean = mean.sort_values()
    std = beta_df[f"{covariate}_std"]
    std = std.loc[mean.index]

    ax.errorbar(
        np.arange(d),
        mean,
        yerr=std,
    )
    ax.axhline(y=0, linestyle="--", color="black")
    ax.set_ylabel(f"log fold change {covariate}")
    ax.set_xticks([])
    ax.set_xlabel("Features")

    return ax
