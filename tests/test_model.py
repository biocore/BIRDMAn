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


class TestModelFit:
    def test_nb_fit(self, table_biom, example_fit):
        beta = example_fit.fit["beta"]
        phi = example_fit.fit["phi"]
        num_cov = len(example_fit.colnames)
        num_chains = example_fit.fit.num_chains
        num_table_feats = len(table_biom.ids(axis="observation"))
        num_draws = example_fit.fit.num_samples * num_chains
        assert beta.shape == (num_cov, num_table_feats - 1, num_draws)
        assert phi.shape == (num_table_feats, num_draws)
