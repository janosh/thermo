"""Data loading and preparation.

Label units (in /data/gaultois_labels.csv):
- electrical resistivity (rho): Ohm * meter
- Seebeck coefficient (S): Volts / Kelvin
- thermal conductivity (kappa): Watts / (meter * Kelvin)
- thermoelectric figure of merit (zT): dimensionless
"""


import numpy as np
import pandas as pd
from pymatgen import MPRester
from pymatgen.ext.cod import COD

from utils import ROOT
from utils.decorators import squeeze


def to_type(df, dtype="float32"):
    """Convert all non-string columns to a different data type.
    E.g. float64 and int to float32.
    """
    df_not_str = df.select_dtypes(exclude=object).astype(dtype)
    df_str = df.select_dtypes(include=object)

    return df_not_str.join(df_str)


def load_gaultois(target_cols: list = ["rho", "seebeck", "kappa", "zT"]):
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


@squeeze
def dropna(*args):
    """Accepts a list of arrays and/or dataframes. Removes all rows containing
    NaNs in the first array/dataframe from each list item.
    """
    # (True|False) mask for each row based on NaN values present in the first dataframe
    mask = ~pd.isnull(args[0])
    if mask.ndim == 2:
        mask = mask.all(1)  # in 2d array, keep row only if all values are not NaN

    return [x.loc[mask] if isinstance(x, pd.DataFrame) else x[mask] for x in args]


@squeeze
def train_test_split(*dfs, test_size: float = 0.1, train=None):
    """Returns training set, test set or both set (split according to test_size)
    depending on train being True, False or None.
    """

    test_index = dfs[0].sample(frac=test_size, random_state=0).index
    mask = dfs[0].index.isin(test_index)

    if train is True:
        return [df.loc[~mask] for df in dfs]

    if train is False:
        return [df.loc[mask] for df in dfs]

    return [df.loc[~mask] for df in dfs], [df.loc[mask] for df in dfs]


def transform_df_cols(
    df, transform="log", cols=["rho", "seebeck_abs", "kappa", "zT", "T"]
):
    if transform == "log":
        # Seebeck coefficient can be negative. Use seebeck_abs when taking the log.
        log_cols = [label + "_log" for label in cols]
        df[log_cols] = np.log(df[cols])
    elif transform == "ihs":
        # ihs: inverse hyperbolic sine.
        ihs_cols = [label + "_ihs" for label in cols]
        df[ihs_cols] = np.arcsinh(df[cols])

    return df


def normalize(df, mean=None, std=None):
    """If mean and std are None, normalize array/dataframe columns to have
    zero mean and unit std. Else use mean and std as provided for normalization.
    """

    # return mean and std to be able to revert normalization later
    mean = mean if mean is not None else df.mean(0)
    std = std if std is not None else df.std(0)

    # ensure we don't divide by zero in columns with zero std (all entries identical)
    try:
        # if df was a 1d array or pd.Series to begin with, std will be a
        # non-subscriptable float, so we handle that case in except
        std[std == 0] = 1
    except TypeError:
        std = std if std > 0 else 1

    return (df - mean) / std, [mean, std]


def fetch_mp():
    """Fetch data from the Materials Project (MP). Docs at
    https://pymatgen.org/_modules/pymatgen/ext/matproj

    Disclaimer: ICSD is a database of materials that actually exist whereas MP has
    all structures where DFT+U converges. Those can be thermo-dynamically unstable
    if they lie above the convex hull.
    """

    # Obtain a Materials Project API key by creating an account at
    # https://materialsproject.org/dashboard.
    API_KEY = "X2UaF2zkPMcFhpnMN"

    # MPRester connects to the Material Project REST interface.
    with MPRester(API_KEY) as mp:
        # mp.query performs the actual API call.
        # The first argument is a dictionary of filter criteria which returned items
        # must satisfy, e.g. criteria = {"material_id": {"$in": ["mp-7988", "mp-69"]}}.
        # Supports all features of the Mongo query syntax.
        # The second argument is a list of quantities of interest which can be
        # selected from https://materialsproject.org/docs/api#Resources_2.
        properties = ["material_id", "pretty_formula", "icsd_ids"]
        # e_above_hull = 0 ensures that all returned materials lie directly on the
        # convex hull and hence are thermodynamically stable.
        criteria = {"e_above_hull": 0}
        data = mp.query(criteria, properties)

    # df[properties] ensures the column order is the same as in the list.
    data = pd.DataFrame(data)[properties].rename(columns={"pretty_formula": "formula"})
    data.to_csv(ROOT + "/data/mp_formulas.csv", index=False, float_format="%g")
    return data


def fetch_cod(formulas=None, ids=None, get_ids_for=None):
    """Fetch data from the Crystallography Open Database (COD).
    Docs at https://pymatgen.org/pymatgen.ext.cod.
    Needs the mysql binary to be in path to run queries. Installable
    via `brew install mysql`.
    """
    cod = COD()
    if formulas:
        return [cod.get_structure_by_formula(f) for f in formulas]
    if ids:
        return [cod.get_structure_by_formula(i) for i in ids]
    if get_ids_for:
        return [cod.get_cod_ids(i) for i in get_ids_for]
    raise ValueError("fetch_cod needs to be passed formulas or ids.")
