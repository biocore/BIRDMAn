from pathlib import Path
from pkg_resources import resource_filename

import biom
from jinja2 import Template
import numpy as np
import pandas as pd
from patsy import dmatrix

from birdman import SingleFeatureModel, TableModel

J2_DIR = Path(resource_filename("birdman", "jinja2"))
SF_TEMPLATE = J2_DIR / "negative_binomial_single.j2.stan"
FULL_TEMPLATE = J2_DIR / "negative_binomial_full.j2.stan"


def create_single_feature_model(
    table: biom.Table,
    metadata: pd.DataFrame,
    stan_file_path: Path,
    fixed_effects: list = None,
    random_effects: list = None,
    beta_prior: float = 5.0,
    group_var_prior: float = 1.0,
    inv_disp_sd_prior: float = 0.5
) -> SingleFeatureModel:
    if not set(table.ids()) == set(metadata.index):
        raise ValueError("Sample IDs must match!")

    fe_formula = " + ".join(fixed_effects)
    dmat = dmatrix(fe_formula, metadata, return_type="dataframe")

    sf_stanfile = _render_stanfile(SF_TEMPLATE, metadata, random_effects)

    with open(stan_file_path, "w") as f:
        f.write(sf_stanfile)

    class _SingleFeatureModel(SingleFeatureModel):
        def __init__(self, feature_id: str):
            super().__init__(table=table, feature_id=feature_id,
                             model_path=stan_file_path)
            self.feature_id = feature_id
            values = table.data(
                id=feature_id,
                axis="observation",
                dense=True
            ).astype(int)

            A = np.log(1 / table.shape[0])

            param_dict = {
                "y": values,
                "x": dmat,
                "p": dmat.shape[1],
                "depth": np.log(table.sum("sample")),
                "A": A,
                "B_p": beta_prior,
                "inv_disp_sd": inv_disp_sd_prior,
                "re_p": group_var_prior
            }

            self.re_dict = dict()

            for group_var in random_effects:
                group_var_series = metadata[group_var].loc[self.sample_names]
                group_subj_map = (
                    group_var_series.astype("category").cat.codes + 1
                )
                param_dict[f"{group_var}_map"] = group_subj_map

                self.re_dict[group_var] = np.sort(group_var_series.unique())

            self.add_parameters(param_dict)

            self.specify_model(
                params=["beta_var", "inv_disp"],
                dims={
                    "beta_var": ["covariate"],
                    "log_lhood": ["tbl_sample"],
                    "y_predict": ["tbl_sample"],
                    "inv_disp": []
                },
                coords={
                    "covariate": dmat.columns,
                    "tbl_sample": self.sample_names,
                },
                include_observed_data=True,
                posterior_predictive="y_predict",
                log_likelihood="log_lhood"
            )

    return _SingleFeatureModel


def create_table_model(
    table: biom.Table,
    metadata: pd.DataFrame,
    stan_file_path: Path,
    fixed_effects: list = None,
    random_effects: list = None,
    beta_prior: float = 5.0,
    group_var_prior: float = 1.0,
    inv_disp_sd_prior: float = 0.5
):
    if not set(table.ids()) == set(metadata.index):
        raise ValueError("Sample IDs must match!")

    fe_formula = " + ".join(fixed_effects)
    dmat = dmatrix(fe_formula, metadata, return_type="dataframe")

    sf_stanfile = _render_stanfile(FULL_TEMPLATE, metadata, random_effects)

    with open(stan_file_path, "w") as f:
        f.write(sf_stanfile)

    class _TableModel(TableModel):
        def __init__(self):
            super().__init__(table=table, model_path=stan_file_path)

            A = np.log(1 / table.shape[0])

            param_dict = {
                "x": dmat,
                "p": dmat.shape[1],
                "depth": np.log(table.sum("sample")),
                "A": A,
                "B_p": beta_prior,
                "inv_disp_sd": inv_disp_sd_prior,
                "re_p": group_var_prior
            }

            self.re_dict = dict()

            for group_var in random_effects:
                group_var_series = metadata[group_var].loc[self.sample_names]
                group_subj_map = (
                    group_var_series.astype("category").cat.codes + 1
                )
                param_dict[f"{group_var}_map"] = group_subj_map

                self.re_dict[group_var] = np.sort(group_var_series.unique())

            self.add_parameters(param_dict)

            self.specify_model(
                params=["beta_var", "inv_disp"],
                dims={
                    "beta_var": ["covariate", "feature_alr"],
                    "log_lhood": ["tbl_sample", "feature"],
                    "y_predict": ["tbl_sample", "feature"],
                    "inv_disp": ["feature"]
                },
                coords={
                    "covariate": dmat.columns,
                    "tbl_sample": self.sample_names,
                    "feature": table.ids("observation"),
                    "feature_alr": table.ids("observation")[1:]
                },
                include_observed_data=True,
                posterior_predictive="y_predict",
                log_likelihood="log_lhood"
            )

    return _TableModel


def _render_stanfile(
    template_path: Path,
    metadata: pd.DataFrame,
    random_effects: list = None
):
    re_dict = dict()
    for group in random_effects:
        n = len(metadata[group].unique())
        re_dict[group] = n

    with open(template_path, "r") as f:
        stanfile = Template(f.read()).render({"re_dict": re_dict})

    return stanfile
