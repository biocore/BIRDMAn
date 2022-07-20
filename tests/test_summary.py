import numpy as np
import pytest

import birdman.summary as summ


@pytest.mark.parametrize("estimator", ["mean", "median", "std"])
def test_summarize_posterior(example_inf, estimator):
    post_summ = summ.summarize_posterior(
        example_inf.posterior,
        "beta_var",
        coords={"covariate": "Intercept"},
        estimator=estimator
    )
    assert len(post_summ) == 27

    exp_index = example_inf.posterior["feature_alr"].to_numpy()
    np.testing.assert_equal(exp_index, post_summ.index)

    post_summ = summ.summarize_posterior(
        example_inf.posterior,
        "beta_var",
        estimator=estimator
    )
    assert post_summ.shape == (27, 2)

    np.testing.assert_equal(exp_index, post_summ.index)
