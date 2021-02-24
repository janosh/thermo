import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr

from thermo.utils import pd2np


def get_err_decay(y_test, y_pred, n_rand=100):
    abs_err = np.abs(y_test - y_pred)
    # increasing count of the number of samples in each element of cumsum()
    n_inc = range(1, len(abs_err) + 1)

    decay_by_err = np.sort(abs_err).cumsum() / n_inc

    # error decay for random exclusion of samples
    ae_tile = np.tile(abs_err, [n_rand, 1])
    [np.random.shuffle(row) for row in ae_tile]  # shuffle rows of ae_tile in place
    rand = ae_tile.cumsum(1) / n_inc
    rand_mean, rand_std = rand.mean(0), rand.std(0)

    return decay_by_err, rand_mean, rand_std


def get_std_decay(y_test, y_pred, y_std):
    abs_err = np.abs(y_test - y_pred)
    y_std_sort = np.argsort(y_std)  # indices that sort y_std in ascending uncertainty
    # increasing count of the number of samples in each element of cumsum()
    n_inc = range(1, len(abs_err) + 1)

    decay_by_std = abs_err[y_std_sort].cumsum() / n_inc

    return decay_by_std


def err_decay(y_test, y_pred, y_stds, title=None, n_rand=100, percentile=True):
    """
    Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error
    """
    xs = range(100 if percentile else len(y_test), 0, -1)

    prctile = lambda seq: np.percentile(seq, xs[::-1])

    if type(y_stds) != dict:
        y_stds = {"std": y_stds}

    for key, y_std in y_stds.items():
        decay_by_std = get_std_decay(y_test, y_pred, y_std)

        if percentile:
            decay_by_std = prctile(decay_by_std)

        plt.plot(xs, decay_by_std, label=key)

    decay_by_err, rand_mean, rand_std = get_err_decay(y_test, y_pred, n_rand)

    if percentile:
        decay_by_err, rand_mean, rand_std = [
            prctile(ys) for ys in [decay_by_err, rand_mean, rand_std]
        ]

    rand_hi, rand_lo = rand_mean + rand_std, rand_mean - rand_std
    plt.plot(xs, decay_by_err, label="error")
    plt.plot(xs[::-1] if percentile else xs, rand_mean)
    plt.fill_between(
        xs[::-1] if percentile else xs, rand_hi, rand_lo, alpha=0.2, label="random"
    )
    plt.ylim([0, rand_mean.mean() * 1.1])

    # n: Number of remaining points in err calculation after discarding the
    # (len(y_test) - n) most uncertain/hightest-error points
    plt.xlabel("error/confidence percentile" if percentile else "excluded samples")
    plt.ylabel("$\\epsilon_\\mathrm{MAE}$")
    plt.title(title)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig


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


def mse_boxes(mse_dfs, x_axis_labels, title=None):
    labels = mse_dfs[0].columns
    # nm_to_mn_cols converts MSEs from being ordered by ML method to
    # being ordered by label (rho, seebeck, ...).
    mse_dfs = nm_to_mn_cols(mse_dfs, x_axis_labels)
    for label, df in zip(labels, mse_dfs):
        # showfliers=False: take outliers into account but don't display them
        ax = sns.boxplot(data=df, width=0.6, showfliers=False)
        ax.set_ylim(0, None)
        for patch in ax.artists:
            *rgb, a = patch.get_facecolor()
            patch.set_facecolor((*rgb, 0.8))

        plt.title(title)
        plt.xlabel("model")
        plt.ylabel("$\\epsilon_\\mathrm{mse}$")

        plt.show()


def show_err_decay_dist(decay_by_std, decay_by_err):
    """Displays the average distance between the decay curves of
    discarding points by uncertainty and discarding points by error.
    Lower is better.
    """
    decay_by_std, decay_by_err = pd2np(decay_by_std, decay_by_err)
    dist = (decay_by_std - decay_by_err).mean()

    text = f"$d = {dist:.2g}$\n"

    rp, _ = pearsonr(decay_by_std, decay_by_err)
    text += f"$r_P = {rp:.2g}$"

    text_box = AnchoredText(text, borderpad=1, loc="upper right", frameon=False)
    plt.gca().add_artist(text_box)


def ci_err_decay(df, n_splits, title=None):
    """Error decay curves with 0.95 confidence intervals."""
    dfs = []
    for df_i in np.array_split(df, n_splits):
        df_i = df_i.reset_index(drop=True)
        decay_by_std = get_std_decay(*df_i.values.T)
        decay_by_err = get_err_decay(*df_i.values.T)
        df_i["decay_by_std"] = decay_by_std
        df_i["decay_by_err"] = decay_by_err
        dfs.append(df_i)

    df = pd.concat(dfs)

    sns.lineplot(data=df[["decay_by_std", "decay_by_err"]])
    show_err_decay_dist(df.decay_by_std, df.decay_by_err)
    plt.gca().invert_xaxis()
    plt.title(title)

    plt.show()
