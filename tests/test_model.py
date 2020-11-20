import pytest

from songbird2.model import Model, NegativeBinomial


class TestModelInheritance:
    def test_nb_model(self, table_df, dmat):
        nb = NegativeBinomial(
            table_df,
            dmat,
            "negative_binomial",
            beta_prior=2.0,
            cauchy_scale=2.0
        )

        assert nb.dat["B_p"] == 2.0
        assert nb.dat["phi_s"] == 2.0
        assert nb.filepath == "templates/negative_binomial.stan"


class TestErrorHandling:
    def test_uncompiled_model_fit(self, table_df, dmat):
        model = Model(table_df, dmat, model_type="uncompiled")

        with pytest.raises(ValueError) as e:
            model.fit_model()

        exp_msg = "Must compile model first!"
        assert str(e.value) == exp_msg

    def test_custom_model_no_filepath(self, table_df, dmat):
        model = Model(table_df, dmat, model_type="custom")

        with pytest.raises(ValueError) as e:
            model.compile_model()

        exp_msg = "Unsupported model type!"
        assert str(e.value) == exp_msg
