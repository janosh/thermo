import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

from utils.decorators import handle_plot

# plt.rcParams.update({"font.size": 14, "font.serif": "Computer Modern Roman"})


def text_box(text, x=0.87, y=0.84, halign="right", valign="top", bbox=None):
    # Add lw=0 to remove the black edge around the bounding box.
    bbox = bbox or dict(facecolor="white", pad=3, lw=1)
    plt.gcf().text(x, y, text, ha=halign, va=valign, bbox=bbox)


def get_corr_p(seq1, seq2):
    [rp, pp], [rs, ps] = [r_fn(seq1, seq2) for r_fn in [pearsonr, spearmanr]]
    return [rp, pp], [rs, ps]


def display_corr(seq1, seq2, prefix="", postfix="", **kwds):
    [rp, _], [rs, _] = get_corr_p(seq1, seq2)
    text = f"$r_\\mathrm{{P}} = {rp:.2g}$\n"
    text += f"$r_\\mathrm{{S}} = {rs:.2g}$"
    text_box(prefix + text + postfix, **kwds)


def display_err_decay_avr_dist(decay_by_std, decay_by_err):
    if isinstance(decay_by_std, (pd.DataFrame, pd.Series)):
        decay_by_std, decay_by_err = (decay_by_std.values, decay_by_err.values)
    dist = np.sum(decay_by_std - decay_by_err) / len(decay_by_std)
    scaler = sklearn.preprocessing.StandardScaler()
    decay_by_std_scd = scaler.fit_transform(decay_by_std.reshape(-1, 1))
    decay_scd_by_err = scaler.transform(decay_by_err.reshape(-1, 1))
    dist_scd = np.sum(decay_by_std_scd - decay_scd_by_err) / len(decay_by_std)
    text = f"$d_\\mathrm{{avg}} = {dist:.2g}$\n"
    text += f"$d_\\mathrm{{norm}} = {dist_scd:.2g}$\n"
    display_corr(decay_by_std, decay_by_err, prefix=text)


@handle_plot
def err_decay(err_name, *args, title=None, bare=False):
    for arg in args:
        countdown = range(len(arg), 0, -1)
        plt.plot(countdown, arg)
    plt.gca().invert_xaxis()
    decay_by_std, decay_by_err = args[:2]
    display_err_decay_avr_dist(decay_by_std, decay_by_err)
    if not bare:
        # n: Number of excluded points starting with points of largest
        # error/uncertainty, resp.
        plt.xlabel("$n$")
        plt.ylabel(f"$\\epsilon_\\mathrm{{{err_name}}}$")
        plt.title(title)


def add_identity(axes, *line_args, **line_kwargs):
    # add parity line (y = x) (add zorder=0 to display data on top of line)
    default_kwargs = dict(alpha=0.3, zorder=0, linestyle="dashed", color="black")
    (identity,) = axes.plot([], [], *line_args, **default_kwargs, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    # Update identity line when moving the plot in interactive
    # viewing mode to always extend to the plot's edges.
    axes.callbacks.connect("xlim_changed", callback)
    axes.callbacks.connect("ylim_changed", callback)
    return axes


@handle_plot
def abs_err_vs_std(abs_err, y_std, title=None, bare=False, log_log=False):
    plt.scatter(abs_err, y_std, s=10)  # s: markersize
    if not bare:
        plt.xlabel("$\\epsilon_\\mathrm{abs} = |y_\\mathrm{true} - y_\\mathrm{pred}|$")
        plt.ylabel("$y_\\mathrm{std}$")
        plt.title(title)
    if log_log:
        plt.yscale("log")
        plt.xscale("log")

    add_identity(plt.gca())
    display_corr(abs_err, y_std)


@handle_plot
def test_vs_pred(y_test, y_pred, y_std=None, title=None, bare=False):
    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    plt.errorbar(y_test, y_pred, yerr=y_std, **styles)
    if not bare:
        plt.xlabel("$y_\\mathrm{test}$")
        plt.ylabel("$y_\\mathrm{pred}$")
        plt.title(title)
    add_identity(plt.gca())
    prefix = f"$\\epsilon_\\mathrm{{mae}} = {np.abs(y_test - y_pred).mean():.3g}$\n"
    display_corr(y_test, y_pred, x=0.15, halign="left", prefix=prefix)


@handle_plot
def std_vs_pred(y_std, y_pred, title=None):
    plt.scatter(y_std, y_pred)
    plt.xlabel("$y_\\mathrm{std}$")
    plt.ylabel("$y_\\mathrm{pred}$")
    plt.title(title)


@handle_plot
def loss_history(loss_histories):
    for name, loss in loss_histories.items():
        plt.plot(loss, label=name)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


@handle_plot
def log_probs(log_probs, text=None, title=None):
    for legend_label, log_prob in log_probs:
        plt.plot(log_prob, label=legend_label)
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.title(title)
    if text:
        plt.gcf().text(*(0.5, -0.1), text, horizontalalignment="center")


@handle_plot
def step_sizes(step_sizes):
    plt.plot(step_sizes)
    plt.xlabel("step")
    plt.ylabel("step size")


@handle_plot
def mse_boxes(df, title=None, bare=False):
    # showfliers=False: Take outliers in the data into account but don't display them.
    ax = sns.boxplot(data=df, width=0.6, showfliers=False)
    ax.set_ylim(0, None)
    for patch in ax.artists:
        *rgb, a = patch.get_facecolor()
        patch.set_facecolor((*rgb, 0.8))
    if not bare:
        plt.title(title)
        plt.xlabel("model")
        plt.ylabel("$\\epsilon_\\mathrm{mse}$")


@handle_plot
def cv_mse_decay(df, title=None, bare=False):
    ax = sns.lineplot(data=df[["err_decay_by_std", "true_err_decay"]])
    display_err_decay_avr_dist(df.err_decay_by_std, df.true_err_decay)
    if title:
        plt.title(title)
    if bare:
        ax.get_legend().remove()
        plt.xlabel("")
        plt.ylabel("")
