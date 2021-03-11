import birdman.diagostics as diag


# TODO: Turn inference transformations into fixture
def test_ess(example_model, example_parallel_model):
    inference = example_model.to_inference_object(
        params=["beta", "phi"],
        coords={
            "feature": example_model.feature_names,
            "covariate": example_model.colnames,
        },
        dims={
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
        },
        alr_params=["beta"],
    )

    inference_parallel = example_parallel_model.to_inference_object(
        params=["beta", "phi"],
        coords={
            "feature": example_model.feature_names,
            "covariate": example_model.colnames,
        },
        dims={
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
        }
    )

    diag.ess(inference, params=["beta", "phi"])
    diag.ess(inference_parallel, params=["beta", "phi"])


def test_rhat(example_model, example_parallel_model):
    inference = example_model.to_inference_object(
        params=["beta", "phi"],
        coords={
            "feature": example_model.feature_names,
            "covariate": example_model.colnames,
        },
        dims={
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
        },
        alr_params=["beta"],
    )

    inference_parallel = example_parallel_model.to_inference_object(
        params=["beta", "phi"],
        coords={
            "feature": example_model.feature_names,
            "covariate": example_model.colnames,
        },
        dims={
            "beta": ["covariate", "feature"],
            "phi": ["feature"],
        }
    )

    diag.rhat(inference, params=["beta", "phi"])
    diag.rhat(inference_parallel, params=["beta", "phi"])
