import numpy as np

from birdman import diagnostics as diag


def test_ppc():
    actual = np.array([
        [1, 3, 0, 0, 5],
        [2, 0, 0, 4, 1],
        [0, 0, 1, 1, 3],
        [3, 2, 1, 2, 0],
    ])
    # [1, 3, 0, 0, 5, 2, 0, 0, 4, 1, 0, 0, 1, 1, 3, 3, 2, 1, 2, 0]
    sort_order = actual.reshape([1, 20]).argsort()

    pred1 = [2, 2, 0, 1, 4, 1, 2, 2, 4, 0, 0, 3, 1, 1, 4, 1, 2, 0, 0, 1]
    pred1 = np.array(pred1)
    pred2 = [1, 1, 1, 0, 6, 3, 4, 0, 0, 3, 1, 2, 0, 1, 7, 0, 1, 1, 0, 0]
    pred2 = np.array(pred2)
    predicted = np.stack(
        [pred1.reshape([4, 5]), pred2.reshape([4, 5])],
        axis=0
    )

    sorted_predictions, sorted_actuals = diag.ppc_values(predicted, actual)

    for i in range(predicted.shape[0]):
        pred_order = sorted_predictions[i]
        expected_pred_order = predicted[i, :, :].reshape(-1)[sort_order][0]

        np.testing.assert_array_equal(pred_order, expected_pred_order)
