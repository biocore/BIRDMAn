import numpy as np

from songbird2 import diagnostics as diag


def test_ppc():
    actual = np.array([
        [1, 3, 0, 0, 5],
        [2, 0, 0, 4, 1],
        [0, 0, 1, 1, 3],
        [3, 2, 1, 2, 0],
    ])
    sort_order = actual.reshape([1, 20]).argsort()
    print(sort_order)

    np.random.seed(42)
    predicted = np.random.randint(low=-2, high=2, size=(3, 4, 5)) + actual
    predicted = predicted.clip(min=0)

    sorted_predictions, sorted_actuals = diag.ppc_values(predicted, actual)

    for i in range(predicted.shape[0]):
        pred_order = sorted_predictions[i]
        expected_pred_order = predicted[i, :, :].reshape(0)[sort_order][0]

        np.testing.assert_array_equal(pred_order, expected_pred_order)
