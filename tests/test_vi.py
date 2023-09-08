from pkg_resources import resource_filename

from birdman import (NegativeBinomial, NegativeBinomialLME,
                     NegativeBinomialSingle, ModelIterator)

def test_nb_fit_vi(table_biom, metadata):
    nb = NegativeBinomial(
        table=table_biom,
        formula="host_common_name",
        metadata=metadata,
        chains=4,
        seed=42,
        num_iter=500,
        beta_prior=2.0,
        inv_disp_sd=0.5,
    )
    nb.compile_model()
    nb.fit_model(method="vi", sampler_args={"output_samples": 100})
    sample_shape = nb.fit.variational_sample.shape
    assert sample_shape[0] == 100


def test_nb_single_fit_vi(table_biom, metadata):
    md = metadata.copy()
    for fid in table_biom.ids(axis="observation"):
        nb = NegativeBinomialSingle(
            table=table_biom,
            feature_id=fid,
            formula="host_common_name",
            metadata=md,
            num_iter=500,
        )
        nb.compile_model()
        nb.fit_model(method="vi",
                     sampler_args={"output_samples": 100, "grad_samples": 20})
        assert nb.fit.variational_sample.shape[0] == 100
