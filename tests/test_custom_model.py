from pkg_resources import resource_filename

import numpy as np

from birdman import Model


def test_custom_model(table_biom, metadata):
    # Negative binomial model with separate prior values for intercept &
    # host_common_name effect and constant overdispersion parameter.
    custom_model = Model(
        table=table_biom,
        formula="host_common_name",
        metadata=metadata,
        model_path=resource_filename("tests", "custom_model.stan"),
        num_iter=100,
        chains=4,
        seed=42,
        parallelize_across="chains"
    )
    custom_model.add_parameters(
        {
            "B_p_1": 2.0,
            "B_p_2": 5.0,
            "phi_s": 0.2
        }
    )
    custom_model.compile_model()
    custom_model.fit_model()

    inference = custom_model.to_inference_object(
        params_to_include=["beta_var"],
        alr_params=["beta_var"]
    )

    assert inference.groups() == ["posterior"]
    ds = inference.posterior

    assert ds.coords._names == {"chain", "covariate", "draw", "feature"}
    assert ds["beta_var"].shape == (2, 28, 4, 100)

    exp_feature_names = table_biom.ids(axis="observation")
    ds_feature_names = ds.coords["feature"]
    assert (exp_feature_names == ds_feature_names).all()

    exp_coord_names = [
        "Intercept",
        "host_common_name[T.long-tailed macaque]"
    ]
    ds_coord_names = ds.coords["covariate"]
    assert (exp_coord_names == ds_coord_names).all()

    assert (ds.coords["draw"] == np.arange(100)).all()
    assert (ds.coords["chain"] == [0, 1, 2, 3]).all()
