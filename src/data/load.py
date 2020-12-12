import pandas as pd

from utils import ROOT


def to_type(df, dtype="float32"):
    """Convert all non-string columns to a different data type.
    E.g. float64 and int to float32.
    """
    df_not_str = df.select_dtypes(exclude=object).astype(dtype)
    df_str = df.select_dtypes(include=object)

    return df_not_str.join(df_str)


def load_gaultois(target_cols: list = ["rho", "seebeck", "kappa", "zT"]):
    """Load Magpie features and targets of the hand-curated
    Gaultois thermoelectrics database.

    Label units (in /data/gaultois_labels.csv):
    - electrical resistivity (rho): Ohm * meter
    - Seebeck coefficient (S): Volts / Kelvin
    - thermal conductivity (kappa): Watts / (meter * Kelvin)
    - thermoelectric figure of merit (zT): dimensionless

    Args:
        target_cols (list, optional): Which targets to load.
        Defaults to ["rho", "seebeck", "kappa", "zT"].

    Returns:
        tuple: 2 dataframes for features and labels
    """
    features = pd.read_csv(ROOT + "/data/gaultois_features.csv").drop(
        columns=["formula"]
    )
    labels = pd.read_csv(ROOT + "/data/gaultois_labels.csv", header=1)

    if target_cols:
        labels = labels[target_cols]

    return to_type(features), to_type(labels)


def load_screen():
    """Load material candidates into a dataframe for screening. Available columns
    are formula, database ID and MagPie features for over 80,000 compositions pulled
    from COD and ICSD.
    """
    features = pd.read_csv(ROOT + "/data/screen_features.csv")
    formulas = pd.read_csv(ROOT + "/data/screen_formulas.csv", comment="#")

    return to_type(formulas), to_type(features)
