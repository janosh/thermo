import numpy as np
import pandas as pd
from pandas import Series


def mse(arr1, arr2, axis=0):
    return np.square(arr1 - arr2).mean(axis)


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


def df_corr(df1, df2, methods=["pearson", "spearman"]):
    return pd.DataFrame([df1.corrwith(df2, method=m) for m in methods], index=methods)


def denorm(mean, std, y_pred, y_var):
    y_pred = y_pred * std + mean
    y_var = std ** 2 * y_var
    return y_pred, y_var


def unlog_preds(y_log_pred, y_log_var):
    y_pred = np.exp(y_log_pred)
    # See side panel of https://wikipedia.org/wiki/Log-normal_distribution
    # for the variance of a log-normally distributed variable.
    y_var = (np.exp(y_log_var) - 1) * np.exp(2 * np.log(y_pred) + y_log_var)
    return y_pred, y_var


def filter_low_risk_high_ret(
    y_pred: Series,
    y_std: Series,
    max_risk: float = 0.25,
    min_ret: float = 1,
    risk_percentile: float = None,
    return_percentile: float = None,
) -> Series.index:
    """Filters a list of model predictions and uncertainties for those with highest return
    (high prediction) and low risk (low uncertainty).

    Args:
        y_pred (Series): Model predictions.
        y_std (Series): Model uncertainties.
        max_risk (float, optional): Cutoff value for model uncertainty. Ignored
            if risk_percentile is not None. Defaults to 0.25.
        min_ret (float, optional): Minimum model prediction. Ignored if
            return_percentile is not None. Defaults to 1.
        risk_percentile (float, optional): Maximum risk percentile. Defaults to None.
        return_percentile (float, optional): Minimum return percentile.
            Defaults to None.

    Returns:
        Series.index: Series index of low-risk high-return predictions.
    """
    if risk_percentile and return_percentile:
        high_return_idx = y_pred > y_pred.quantile(return_percentile)
        low_risk_idx = y_std < y_std.quantile(risk_percentile)
        lrhr_mask = np.logical_and(high_return_idx, low_risk_idx)
        y_pred = y_pred[lrhr_mask]
        return y_pred.index

    if risk_percentile:
        y_pred = y_pred[y_std < y_std.quantile(risk_percentile)]
    else:
        y_pred = y_pred[y_std < max_risk]

    if return_percentile:
        y_pred = y_pred[y_pred > y_pred.quantile(return_percentile)]
    else:
        y_pred = y_pred[y_pred > min_ret]

    return y_pred.index
