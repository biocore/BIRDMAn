import numpy as np


def ppc_values(
    predicted: np.ndarray,
    actual: np.ndarray,
):
    """Compute posterior predictive checks.

    Parameters:
    -----------
    predicted (np.ndarray)
        table of predicted feature counts per iteration (iterations x samples x
        features)
    actual (np.ndarray)
        table of "true" counts (samples x features)
    """
    if predicted.shape[1:] != actual.shape:
        raise ValueError(
            "Arrays must have the same number of samples/features!"
        )
    num_it, num_samp, num_feat = predicted.shape

    predicted_flat = np.reshape(predicted, (num_it, -1))
    actual_flat = np.reshape(actual, -1)

    # sort values by actual counts
    sort_order = np.argsort(actual_flat)
    sorted_values = np.apply_along_axis(lambda x: x[sort_order], 1,
                                        predicted_flat)

    return sorted_values, actual_flat[sort_order]
