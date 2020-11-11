from songbird2 import fit_model


class TestModel:
    def test_fit_model(self, data_table, metadata):
        fit_model(
            data_table,
            metadata,
            "body_site",
            num_iter=500,
            seed=42,
        )
