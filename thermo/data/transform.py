"""Data loading and preparation.

Target units (in /data/gaultois_targets.csv):
- electrical resistivity (rho): Ohm * meter
- Seebeck coefficient (S): Volts / Kelvin
- thermal conductivity (kappa): Watts / (meter * Kelvin)
- thermoelectric figure of merit (zT): dimensionless
"""


import numpy as np
import pandas as pd

from thermo.utils.decorators import squeeze


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
    df, transform="log", cols=("rho", "seebeck_abs", "kappa", "zT", "T")
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
    if mean is None:
        mean = df.mean(0)
    if std is None:
        std = df.std(0)

    # ensure we don't divide by zero in columns with zero std (all entries identical)
    try:
        # if df was a 1d array or pd.Series to begin with, std will be a
        # non-subscriptable float, so we handle that case in except
        std[std == 0] = 1
    except TypeError:
        std = std if std > 0 else 1

    # return mean and std to be able to revert normalization later
    return (df - mean) / std, [mean, std]
