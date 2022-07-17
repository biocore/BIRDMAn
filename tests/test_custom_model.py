from pkg_resources import resource_filename

import numpy as np

from birdman import TableModel
from birdman.transform import posterior_alr_to_clr


def test_custom_model(table_biom, metadata):
    # Negative binomial model with separate prior values for intercept &
    # host_common_name effect and constant overdispersion parameter.
    custom_model = TableModel(
        table=table_biom,
        model_path=resource_filename("tests", "custom_model.stan"),
        num_iter=100,
        chains=4,
        seed=42,
    )
    custom_model.create_regression(
        formula="host_common_name",
        metadata=metadata,
    )
    custom_model.add_parameters(
        {
            "B_p_1": 2.0,
            "B_p_2": 5.0,
            "phi_s": 0.2,
            "depth": np.log(table_biom.sum(axis="sample")),
        }
    )
    custom_model.specify_model(
        params=["beta_var"],
        coords={
            "feature": custom_model.feature_names,
            "feature_alr": custom_model.feature_names[1:],
            "covariate": custom_model.colnames
        },
        dims={
            "beta_var": ["covariate", "feature_alr"],
            "phi": ["feature"]
        },
    )
    custom_model.compile_model()
    custom_model.fit_model()
    inference = custom_model.to_inference()

    assert set(inference.groups()) == {"posterior", "sample_stats"}
    ds = inference.posterior

    assert ds.coords._names == {"chain", "covariate", "draw", "feature_alr"}
    assert set(ds["beta_var"].shape) == {2, 27, 4, 100}

    exp_feature_names = table_biom.ids(axis="observation")[1:]
    ds_feature_names = ds.coords["feature_alr"]
    assert (exp_feature_names == ds_feature_names).all()

    exp_coord_names = [
        "Intercept",
        "host_common_name[T.long-tailed macaque]"
    ]
    ds_coord_names = ds.coords["covariate"]
    assert (exp_coord_names == ds_coord_names).all()

    assert (ds.coords["draw"] == np.arange(100)).all()
    assert (ds.coords["chain"] == [0, 1, 2, 3]).all()

    inference.posterior = posterior_alr_to_clr(
        posterior=inference.posterior,
        alr_params=["beta_var"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=custom_model.feature_names
    )

    ds = inference.posterior

    assert ds.coords._names == {"chain", "covariate", "draw", "feature"}
    assert set(ds["beta_var"].shape) == {2, 28, 4, 100}

    exp_feature_names = table_biom.ids(axis="observation")
    ds_feature_names = ds.coords["feature"]
    assert (exp_feature_names == ds_feature_names).all()
