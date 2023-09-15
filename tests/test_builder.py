import logging
from pathlib import Path
import tempfile

import numpy as np
from birdman.builder import create_single_feature_model, create_table_model


def test_create_sf_model(table_biom, metadata):
    rng = np.random.default_rng(42)

    metadata = metadata.copy()
    N = metadata.shape[0]

    # Add pseudo-random effects
    rand_eff_1 = rng.choice([1, 2, 3], N)
    rand_eff_2 = rng.choice([1, 2, 3, 5], N)
    metadata["group_1"] = rand_eff_1
    metadata["group_2"] = rand_eff_2

    with tempfile.TemporaryDirectory() as f:
        stan_file_path = Path(f) / "model.stan"

        BuiltModel = create_single_feature_model(
            table_biom,
            metadata,
            fixed_effects=["host_common_name"],
            random_effects=["group_1", "group_2"],
            stan_file_path=stan_file_path
        )

        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_logger.disabled = True

        first_feat = table_biom.ids("observation")[0]
        model = BuiltModel(first_feat)
        model.compile_model()
        model.fit_model(method="vi")

        inf = model.to_inference()

    post = inf.posterior
    data_vars = set(post.data_vars.keys())
    assert data_vars == {"beta_var", "inv_disp"}

    exp_covariates = ("Intercept", "host_common_name[T.long-tailed macaque]")
    assert (post.coords["covariate"] == exp_covariates).all()

    inf_groups = set(inf.groups())
    assert inf_groups == {"posterior", "posterior_predictive",
                          "log_likelihood", "observed_data"}

    group_1_eff = model.fit.stan_variable("group_1_eff")
    group_2_eff = model.fit.stan_variable("group_2_eff")

    assert group_1_eff.shape == (500, 3)
    assert group_2_eff.shape == (500, 4)


def test_create_full_model(table_biom, metadata):
    rng = np.random.default_rng(42)

    metadata = metadata.copy()
    N = metadata.shape[0]

    # Add pseudo-random effects
    rand_eff_1 = rng.choice([1, 2, 3], N)
    rand_eff_2 = rng.choice([1, 2, 3, 5], N)
    metadata["group_1"] = rand_eff_1
    metadata["group_2"] = rand_eff_2

    with tempfile.TemporaryDirectory() as f:
        stan_file_path = Path(f) / "model.stan"

        BuiltModel = create_table_model(
            table_biom,
            metadata,
            fixed_effects=["host_common_name"],
            random_effects=["group_1", "group_2"],
            stan_file_path=stan_file_path
        )

        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_logger.disabled = True

        model = BuiltModel()
        model.compile_model()
        model.fit_model(method="vi")

        inf = model.to_inference()

    post = inf.posterior
    data_vars = set(post.data_vars.keys())
    assert data_vars == {"beta_var", "inv_disp"}

    exp_covariates = ("Intercept", "host_common_name[T.long-tailed macaque]")
    assert (post.coords["covariate"] == exp_covariates).all()

    inf_groups = set(inf.groups())
    assert inf_groups == {"posterior", "posterior_predictive",
                          "log_likelihood", "observed_data"}

    group_1_eff = model.fit.stan_variable("group_1_eff")
    group_2_eff = model.fit.stan_variable("group_2_eff")

    assert group_1_eff.shape == (500, 3, 27)
    assert group_2_eff.shape == (500, 4, 27)
