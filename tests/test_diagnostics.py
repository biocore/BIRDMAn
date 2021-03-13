import birdman.diagnostics as diag


def test_ess(example_inf, example_parallel_inf):
    diag.ess(example_inf, params=["beta", "phi"])
    diag.ess(example_parallel_inf, params=["beta", "phi"])


def test_rhat(example_inf, example_parallel_inf):
    diag.rhat(example_inf, params=["beta", "phi"])
    diag.rhat(example_parallel_inf, params=["beta", "phi"])


def test_r2_score(example_inf, example_parallel_inf):
    diag.r2_score(example_inf)
    diag.r2_score(example_parallel_inf)


def test_loo(example_inf, example_parallel_inf):
    diag.loo(example_inf)
    diag.loo(example_parallel_inf)
