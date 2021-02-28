"""Functions only needed once to generate CSV files."""

import pandas as pd
from matminer.featurizers import composition as cf
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.conversions import StrToComposition

from thermo.utils import ROOT


def featurize_with_magpy(df, input_col_name="formula", retain_cols=["T"]):
    # Convert the formula string into a composition object to be used by matminer.
    str_to_comp = StrToComposition(target_col_id="composition_obj")
    df = str_to_comp.featurize_dataframe(df, input_col_name)

    # Create featurizers one of which uses the MAGPIE process.
    featurizers = [
        cf.Stoichiometry(),
        cf.ElementProperty.from_preset("magpie"),
        cf.ValenceOrbital(props=["avg"]),
        cf.IonProperty(fast=True),
    ]
    multi_featurizer = MultipleFeaturizer(featurizers)

    # Generate the features. They will be appended as columns to a copy
    # of the input dataframe.
    features = multi_featurizer.featurize_dataframe(df, col_id="composition_obj")

    # Remove columns that were copied over from input dataframe. By default, retain
    # only the input for multi_featurizer (as a reference, not as an actual feature;
    # don't forget to drop this column before feeding the features into a model.)
    feature_columns = [input_col_name] + retain_cols + multi_featurizer.feature_labels()
    features = features[feature_columns]

    return features


def generate_gaultois_features():
    gaultois_targets = pd.read_csv(ROOT + "/data/gaultois_targets.csv", comment="#")
    gaultois_features = featurize_with_magpy(gaultois_targets)
    gaultois_features.to_csv(
        ROOT + "/data/gaultois_magpie_features.csv", index=False, float_format="%g"
    )


def generate_screen_features():
    screen_formulas = pd.read_csv(ROOT + "/data/screen_formulas.csv", comment="#")
    screen_features = featurize_with_magpy(screen_formulas, retain_cols=[])
    screen_features.to_csv(
        ROOT + "/data/screen_set_magpie_features.csv", index=False, float_format="%g"
    )
