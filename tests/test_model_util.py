from songbird2.model import Model
import songbird2.model_util as mutil


def test_collapse_results(ex_model, table_df, dmat):
    model = Model(table_df, dmat, model_type="test")
    model.fit = ex_model["fit"]

    param_df = mutil.collapse_param(model, "beta")
    body_sites = ["left palm", "right palm", "tongue"]  # gut is reference
    body_sites = [f"body_site[T.{x}]" for x in body_sites]
    levels = ["Intercept"] + body_sites
    mean_cols = [x + "_mean" for x in levels]
    std_cols = [x + "_std" for x in levels]
    all_cols = set(mean_cols).union(set(std_cols))

    assert set(param_df.columns) == all_cols
