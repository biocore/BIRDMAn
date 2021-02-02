from birdman import NegativeBinomial


class TestModelInheritance:
    def test_nb_model(self, table_biom, metadata):
        nb = NegativeBinomial(
            table_biom,
            "host_common_name",
            metadata,
            "negative_binomial",
            beta_prior=2.0,
            cauchy_scale=2.0
        )

        assert nb.dat["B_p"] == 2.0
        assert nb.dat["phi_s"] == 2.0
        assert nb.filepath == "templates/negative_binomial.stan"
