import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatviz as pmv
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr


def plot_output(y_test, y_pred, y_std=None, **kwargs):
    """Convenience function for generating multiple plots in one go for
    analyzing a model's accuracy and quality of uncertainty estimates.
    """
    fig1 = pmv.density_scatter(x=y_test, y=y_pred, **kwargs)

    if y_std is None:
        return fig1

    fig2 = pmv.error_decay_with_uncert(y_test, y_pred, y_std, **kwargs)

    abs_err = abs(y_test - y_pred)
    fig3 = pmv.density_scatter(
        x=abs_err,
        y=y_std,
        xlabel="Absolute error",
        ylabel="Model uncertainty",
        **kwargs,
    )
    return fig1, fig2, fig3


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


def nm_to_mn_cols(dfs, keys):
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


def mse_boxes(
    mse_dfs: list[pd.DataFrame], x_axis_labels: list[str], title: str | None = None
) -> None:
    """Plot MSEs of multiple models."""
    # nm_to_mn_cols converts MSEs from being ordered by ML method to
    # being ordered by label (rho, seebeck, ...).
    mse_dfs = nm_to_mn_cols(mse_dfs, x_axis_labels)
    for df in mse_dfs:
        # showfliers=False: take outliers into account but don't display them
        ax = sns.boxplot(data=df, width=0.6, showfliers=False)
        ax.set_ylim(0, None)
        for patch in ax.artists:
            *rgb, _alpha = patch.get_facecolor()
            patch.set_facecolor((*rgb, 0.8))

        plt.title(title)
        plt.xlabel("model")
        plt.ylabel("$\\epsilon_\\mathrm{mse}$")

        plt.show()


def ci_err_decay(df, n_splits, title=None):
    """Error decay curves with 0.95 confidence intervals."""
    dfs = []
    for df_i in np.array_split(df, n_splits):
        df_i = df_i.reset_index(drop=True)
        y_true, y_pred, y_std = df_i.values.T
        abs_err = np.abs(y_true - y_pred)
        n_samples = len(abs_err)

        # Calculate decay by uncertainty (std)
        decay_by_std = abs_err[np.argsort(y_std)].cumsum() / np.arange(1, n_samples + 1)

        # Calculate optimal decay by error
        decay_by_err = np.sort(abs_err).cumsum() / np.arange(1, n_samples + 1)

        df_i["decay_by_std"] = decay_by_std
        df_i["decay_by_err"] = decay_by_err
        dfs += [df_i]

    df = pd.concat(dfs)

    sns.lineplot(data=df[["decay_by_std", "decay_by_err"]])

    dist = (df.decay_by_std - df.decay_by_err).mean()

    text = f"$d = {dist:.2g}$\n"

    rp, _ = pearsonr(df.decay_by_std, df.decay_by_err)
    text += f"$r_P = {rp:.2g}$"

    text_box = AnchoredText(text, borderpad=1, loc="upper right", frameon=False)
    plt.gca().add_artist(text_box)

    plt.gca().invert_xaxis()
    plt.title(title)

    plt.show()
