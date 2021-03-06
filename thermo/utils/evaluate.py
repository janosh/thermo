import numpy as np
import pandas as pd

from thermo import plots


def mse(arr1, arr2, axis=0):
    return ((arr1 - arr2) ** 2).mean(axis)


def rmse(arr1, arr2, axis=0):
    return np.sqrt(mse(arr1, arr2, axis))


def mae(arr1, arr2, axis=0):
    return abs(arr1 - arr2).mean(axis)


def compute_zT(rho, seebeck, kappa, temperature):
    """Returns the thermoelectric figure of merit.
    Note: All inputs should have SI units.
    """
    return seebeck ** 2 * temperature / (rho * kappa)


def compute_log_zT(log_rho, log_seebeck, log_kappa, log_temperature):
    """Returns the log of the thermoelectric figure of merit."""
    return 2 * log_seebeck + log_temperature - log_rho - log_kappa


def compute_log_zT_var(log_rho_var, log_seebeck_sqr_var, log_kappa_var):
    """Compute the variance of the logarithmic thermoelectric figure
    of merit zT.
    """
    return log_rho_var + log_seebeck_sqr_var + log_kappa_var


def get_df_stats(df):
    """Return column-wise statistics (mean, median, standard deviation,
    min, max) of a dataframe. Useful for quick sanity checks.
    """
    stats = pd.concat([df.mean(), df.median(), df.std(), df.min(), df.max()], axis=1).T
    stats.index = ["mean", "median", "std", "min", "max"]
    stats.columns = df.columns
    return stats


def dfs_have_same_col_names(dfs, sort=False):
    """Returns True when a list of dataframes have the same column names
    in the same order. Pass sort=True if order doesn't matter.

    Args:
        dfs (list): List of dataframes.
        sort (bool, optional): Whether to sort the columns before comparing.
            Defaults to False.
    """
    if sort:
        return np.all([sorted(dfs[0].columns) == sorted(i.columns) for i in dfs])
    return np.all([list(dfs[0].columns) == list(i.columns) for i in dfs])


def nxm_to_mxn_cols(dfs, keys):
    """Creates m n-column dataframes from n m-column dataframes.
    Adapted from https://stackoverflow.com/a/57339017.

    Args:
        dfs: List of n m-column dataframes. Must all have the same column names!
        keys: List of n column names for the new dataframes.
    """
    assert dfs_have_same_col_names(dfs), "Unequal column names in passed dataframes."
    df_concat = pd.concat(dfs, keys=keys).unstack(0)
    mxn_dfs = [df_concat.xs(i, axis=1, level=0) for i in df_concat.columns.levels[0]]
    for i, df in enumerate(mxn_dfs):
        df.name = df.columns.name = dfs[0].columns[i]
    return mxn_dfs


def mse_boxes(mse_dfs, x_axis_labels, **kwargs):
    labels = mse_dfs[0].columns
    # nxm_to_mxn_cols converts MSEs from being ordered by ML method to
    # being ordered by label (rho, seebeck, ...).
    mse_dfs = nxm_to_mxn_cols(mse_dfs, x_axis_labels)
    for label, df in zip(labels, mse_dfs):
        plots.mse_boxes(df, label, **kwargs)


def df_corr(df1, df2, methods=["pearson", "spearman"]):
    return pd.DataFrame([df1.corrwith(df2, method=m) for m in methods], index=methods)


def back_transform_targets(mean, std, y_pred, y_var, to="log"):
    y_pred = y_pred * std + mean
    y_var = std ** 2 * y_var
    if to == "orig":
        y_pred = np.exp(y_pred)
        # See side panel of https://wikipedia.org/wiki/Log-normal_distribution
        # for the variance of a log-normally distributed variable.
        y_var = (np.exp(y_var) - 1) * np.exp(2 * np.log(y_pred) + y_var)
    elif to != "log":
        raise ValueError(f"`to` must be log or orig, got {to}")
    return y_pred, y_var


def filter_low_risk_high_ret(df, max_risk=0.25, min_ret=1, cols=["zT_pred", "zT_var"]):
    y_pred, y_var = df[cols].to_numpy().T
    y_pred_low_risk = y_pred[y_var < max_risk]
    df = df[y_var < max_risk]
    df = df[y_pred_low_risk > min_ret]
    return df
