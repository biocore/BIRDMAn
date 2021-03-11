import birdman.diagostics as diag


def test_ess(example_inf, example_parallel_inf):
    diag.ess(example_inf, params=["beta", "phi"])
    diag.ess(example_parallel_inf, params=["beta", "phi"])


def test_rhat(example_inf, example_parallel_inf):
    diag.rhat(example_inf, params=["beta", "phi"])
    diag.rhat(example_parallel_inf, params=["beta", "phi"])
