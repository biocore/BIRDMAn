import numpy as np

from birdman import transform


def test_inference_alr_to_clr(example_model):
    inf = example_model.to_inference_object()
    np.testing.assert_equal(
        inf.posterior["beta"].coords["feature_alr"],
        example_model.feature_names[1:],
    )

    new_post = transform.inference_alr_to_clr(
        inf.posterior,
        alr_params=["beta"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=example_model.feature_names
    )

    assert set(new_post.dims) == {"chain", "draw", "feature", "covariate"}
    np.testing.assert_equal(
        new_post.coords["feature"],
        example_model.feature_names
    )
