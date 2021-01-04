import pandas as pd

from thermo.utils import ROOT


def to_type(df, dtype="float32"):
    """Convert all non-string columns to a different data type.
    E.g. float64 and int to float32.
    """
    df_not_str = df.select_dtypes(exclude=object).astype(dtype)
    df_str = df.select_dtypes(include=object)

    return df_not_str.join(df_str)


def load_gaultois(
    target_cols: list = ["rho", "seebeck", "kappa", "zT"], drop_outliers=False
):
    """Load Magpie features and targets of the hand-curated
    Gaultois thermoelectrics database.

    Label units (in /data/gaultois_targets.csv):
    - electrical resistivity (rho): Ohm * meter
    - Seebeck coefficient (S): Volts / Kelvin
    - thermal conductivity (kappa): Watts / (meter * Kelvin)
    - thermoelectric figure of merit (zT): dimensionless

    Args:
        target_cols (list, optional): Which targets to load.
        Defaults to ["rho", "seebeck", "kappa", "zT"].

    Returns:
        tuple: 2 dataframes for features and targets
    """
    features = pd.read_csv(ROOT + "/data/gaultois_features.csv").drop(
        columns=["formula"]
    )
    targets = pd.read_csv(ROOT + "/data/gaultois_targets.csv", header=1)

    if drop_outliers:
        features = features[targets.outliers.isna()]
        targets = targets[targets.outliers.isna()]

    if target_cols:
        targets = targets[target_cols]

    return to_type(features), to_type(targets)


def load_screen():
    """Load material candidates into a dataframe for screening. Available columns
    are formula, database ID and MagPie features for over 80,000 compositions pulled
    from COD and ICSD.
    """
    features = pd.read_csv(ROOT + "/data/screen_features.csv")
    formulas = pd.read_csv(ROOT + "/data/screen_formulas.csv", comment="#")

    return to_type(formulas), to_type(features)
