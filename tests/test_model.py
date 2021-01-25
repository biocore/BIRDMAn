import pytest

from birdman.model import Model, NegativeBinomial


class TestModelInheritance:
    def test_nb_model(self, table_biom, metadata):
        nb = NegativeBinomial(
            table_biom,
            "body_site",
            metadata,
            "negative_binomial",
            beta_prior=2.0,
            cauchy_scale=2.0
        )

        assert nb.dat["B_p"] == 2.0
        assert nb.dat["phi_s"] == 2.0
        assert nb.filepath == "templates/negative_binomial.stan"


class TestErrorHandling:
    def test_uncompiled_model_fit(self, table_biom, metadata):
        model = Model(table_biom, "body_site", metadata,
                      model_type="uncompiled")

        with pytest.raises(ValueError) as e:
            model.fit_model()

        exp_msg = "Must compile model first!"
        assert str(e.value) == exp_msg

    def test_custom_model_no_filepath(self, table_biom, metadata):
        model = Model(table_biom, "body_site", metadata, model_type="custom")

        with pytest.raises(ValueError) as e:
            model.compile_model()

        exp_msg = "Unsupported model type!"
        assert str(e.value) == exp_msg
