import numpy as np
import pandas as pd

from utils import plots


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
    """Returns the log of the thermoelectric figure of merit.
    """
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


def get_err_decay(y_true, y_pred, y_std, err_func=mse):
    # Get indices that sort y_std in ascending uncertainty.
    y_std_sort = np.argsort(y_std)
    # Get indices that sort y_pred in ascending absolute error.
    y_err_sort = np.argsort(np.abs(y_pred - y_true))

    decay_by_std, decay_by_err = [], []
    # Loop over [n, n-1, ..., 2, 1].
    for idx in range(y_true.size, 0, -1):
        # Compute error (e.g. MSE) with the n = (y_true.size - idx) most uncertain
        # points excluded.
        less_y_std = y_std_sort[:idx]
        decay_by_std.append(err_func(y_pred[less_y_std], y_true[less_y_std]))

        # Compute error with the n = (y_true.size - idx) largest error points excluded.
        less_err = y_err_sort[:idx]
        decay_by_err.append(err_func(y_pred[less_err], y_true[less_err]))
    return np.array([decay_by_std, decay_by_err])


def plot_output(y_test, y_pred, y_std=None, **kwargs):
    """Convenience function for generating multiple plots in one go for
    analyzing a model's accuracy and quality of uncertainty estimates.
    """
    plots.true_vs_pred(y_test, y_pred, y_std=y_std, **kwargs)
    if y_std is None:
        return

    decay_by_std, decay_by_err = get_err_decay(y_test, y_pred, y_std, mse)
    plots.err_decay("mse", decay_by_std, decay_by_err, **kwargs)

    abs_err = abs(y_test - y_pred)
    plots.abs_err_vs_std(abs_err, y_std, **kwargs)


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


def ci_mse_decay_plots(y_test, y_pred, y_var, **kwargs):
    """MSE decay/drop-off curves with confidence intervals.
    """
    rho, seebeck_abs, kappa, zT = nxm_to_mxn_cols(
        [y_test, y_pred, y_var], keys=["y_test", "y_pred", "y_var"]
    )
    for df in [rho, seebeck_abs, kappa, zT]:
        for col, values in zip(
            ["err_decay_by_std", "true_err_decay"],
            zip(*[get_err_decay(*arr.values.T) for arr in pd.np.array_split(df, 5)]),
        ):
            df[col] = pd.np.concatenate(values)
        plots.cv_mse_decay(df, **kwargs)


def r2_score(true, pred):
    # https://wikipedia.org/wiki/Coefficient_of_determination
    true, pred = np.array([true, pred])
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def back_transform_labels(mean, std, y_pred, y_var, to="log"):
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
