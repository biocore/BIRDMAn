import birdman.diagnostics as diag


def test_ess(example_inf):
    diag.ess(example_inf, params=["beta", "phi"])


def test_rhat(example_inf):
    diag.rhat(example_inf, params=["beta", "phi"])


def test_r2_score(example_inf):
    diag.r2_score(example_inf)


def test_loo(example_inf):
    diag.loo(example_inf)
