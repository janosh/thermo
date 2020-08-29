import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import YlGn
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from pymatgen.core import Composition
from scipy.stats import gaussian_kde, pearsonr

from utils import ROOT, pd2np


def add_text_box(text):
    # Add lw=0 to remove the black edge around the bounding box.
    # prop: keyword params passed to the Text instance inside AnchoredText.
    prop = {"bbox": {"lw": 0.5, "facecolor": "white"}}
    text_box = AnchoredText(text, borderpad=1, prop=prop, loc="upper right", pad=0.2)
    plt.gca().add_artist(text_box)


def corr_text(seq1, seq2):
    rp, _ = pearsonr(seq1, seq2)
    text = f"$r_P = {rp:.2g}$"
    return text


def show_err_decay_dist(decay_by_std, decay_by_err):
    """Displays the average distance between the decay curves of
    discarding points by uncertainty and discarding points by error.
    Lower is better.
    """
    decay_by_std, decay_by_err = pd2np(decay_by_std, decay_by_err)
    dist = (decay_by_std - decay_by_err).mean()

    text = f"$d = {dist:.2g}$\n"
    text += corr_text(decay_by_std, decay_by_err)
    add_text_box(text)


def err_decay(err_name, decay_by_std, decay_by_err, title=None):
    """Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error

    Args:
        err_name (str): usually MAE or MSE
        title (str, optional): plot title
    """
    countdown = range(len(decay_by_std), 0, -1)
    plt.plot(countdown, decay_by_std)
    plt.plot(countdown, decay_by_err)

    plt.gca().invert_xaxis()
    show_err_decay_dist(decay_by_std, decay_by_err)

    # n: Number of excluded points starting with points of largest
    # error/uncertainty, resp.
    plt.xlabel("$n$")
    plt.ylabel(f"$\\epsilon_\\mathrm{{{err_name}}}$")
    plt.title(title)

    fig = plt.gcf()
    plt.show()
    return fig


def add_identity(axis, **line_kwargs):
    """Add a parity line (y = x) (aka identity) to the provided axis.
    """
    # zorder=0 ensures other plotted data displays on top of line
    default_kwargs = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
    (identity,) = axis.plot([], [], **default_kwargs, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axis)
    # Update identity line when moving the plot in interactive
    # viewing mode to always extend to the plot's edges.
    axis.callbacks.connect("xlim_changed", callback)
    axis.callbacks.connect("ylim_changed", callback)
    return axis


def abs_err_vs_std(abs_err, y_std, title=None):
    plt.scatter(abs_err, y_std, s=10)  # s: markersize

    plt.xlabel("$\\epsilon_\\mathrm{abs} = |y_\\mathrm{true} - y_\\mathrm{pred}|$")
    plt.ylabel("$y_\\mathrm{std}$")
    plt.title(title)

    # plt.axis("scaled")
    add_identity(plt.gca())
    text = corr_text(abs_err, y_std)
    add_text_box(text)

    fig = plt.gcf()
    plt.show()
    return fig


def true_vs_pred(y_test, y_pred, y_std=None, title=None):
    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    plt.errorbar(y_test, y_pred, yerr=y_std, **styles)

    plt.xlabel("$y_\\mathrm{test}$")
    plt.ylabel("$y_\\mathrm{pred}$")
    plt.title(title)

    add_identity(plt.gca())
    text = f"$\\epsilon_\\mathrm{{mae}} = {np.abs(y_test - y_pred).mean():.3g}$\n"

    text += corr_text(y_test, y_pred)
    add_text_box(text)

    fig = plt.gcf()
    plt.show()
    return fig


def std_vs_pred(y_std, y_pred, title=None):
    plt.scatter(y_std, y_pred)
    plt.xlabel("$y_\\mathrm{std}$")
    plt.ylabel("$y_\\mathrm{pred}$")
    plt.title(title)

    plt.show()


def loss_history(hist):
    fig = plt.figure(figsize=[12, 5])
    for key, data in hist.items():
        if "loss" in key:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(data, label=key)
            ax1.set(xlabel="epoch")
        else:  # plot other metrics like accuracy or loss without
            # regularizers on a separate axis
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(data, label=key)
            ax2.set(xlabel="epoch")

    [ax.legend() for ax in fig.axes]

    plt.show()


def log_probs(log_probs, text=None, title=None):
    for legend_label, log_prob in log_probs:
        plt.plot(log_prob, label=legend_label)
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.title(title)
    if text:
        plt.gcf().text(*(0.5, -0.1), text, horizontalalignment="center")

    plt.show()


def step_sizes(step_sizes):
    plt.plot(step_sizes)
    plt.xlabel("step")
    plt.ylabel("step size")

    plt.show()


def mse_boxes(df, title=None):
    # showfliers=False: Take outliers in the data into account but don't display them.
    ax = sns.boxplot(data=df, width=0.6, showfliers=False)
    ax.set_ylim(0, None)
    for patch in ax.artists:
        *rgb, a = patch.get_facecolor()
        patch.set_facecolor((*rgb, 0.8))

    plt.title(title)
    plt.xlabel("model")
    plt.ylabel("$\\epsilon_\\mathrm{mse}$")

    plt.show()


def cv_mse_decay(df, title=None):
    sns.lineplot(data=df[["err_decay_by_std", "true_err_decay"]])
    show_err_decay_dist(df.err_decay_by_std, df.true_err_decay)
    if title:
        plt.title(title)

    plt.show()


def count_elements(formulas):
    """Count occurrences of each element in a materials dataset.

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]

    Returns:
        pd.Series: Number of appearances for each element in formulas.
    """

    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")
    elem_tracker = pd.Series(0, index=ptable.symbol)  # symbols = [H, He, Li, etc.]

    for formula in formulas:
        formula_dict = Composition(formula).as_dict()
        elem_count = pd.Series(formula_dict, name="count")
        elem_tracker = elem_tracker.add(elem_count, fill_value=0)

    return elem_tracker, ptable


def ptable_elemental_prevalence(formulas, log_scale=False):
    """Colormap the periodic table according to the prevalence of each element
    in a materials dataset.
    Adapted from https://github.com/kaaiian/ML_figures.

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        log_scale (bool, optional): Whether color map scale is log or linear.
    """
    elem_tracker, ptable = count_elements(formulas)

    n_row = ptable.row.max()
    n_column = ptable.column.max()

    plt.figure(figsize=(n_column, n_row))

    rw = rh = 0.9  # rectangle width/height
    count_min = elem_tracker.min()
    count_max = elem_tracker.max()

    norm = Normalize(
        vmin=0 if log_scale else count_min,
        vmax=np.log(count_max) if log_scale else count_max,
    )

    text_style = dict(
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=20,
        fontweight="semibold",
        color="black",
    )

    for symbol, row, column, _ in ptable.values:
        row = n_row - row
        count = elem_tracker[symbol]
        if log_scale and count != 0:
            count = np.log(count)
        color = YlGn(norm(count)) if count != 0 else "silver"

        if row < 3:
            row += 0.5
        rect = Rectangle((column, row), rw, rh, edgecolor="gray", facecolor=color)

        plt.text(column + rw / 2, row + rw / 2, symbol, **text_style)

        plt.gca().add_patch(rect)

    granularity = 20
    x_offset = 3.5
    y_offset = 7.8
    length = 9
    for i in range(granularity):
        value = int(round((i) * count_max / (granularity - 1)))
        if log_scale and value != 0:
            value = np.log(value)
        color = YlGn(norm(value)) if value != 0 else "silver"
        x_loc = i / (granularity) * length + x_offset
        width = length / granularity
        height = 0.35
        rect = Rectangle(
            (x_loc, y_offset), width, height, edgecolor="gray", facecolor=color
        )

        if i in [0, 4, 9, 14, 19]:
            text = f"{value:.0g}"
            if log_scale:
                text = f"{np.exp(value):.0g}".replace("e+0", "e")
            plt.text(x_loc + width / 2, y_offset - 0.4, text, **text_style)

        plt.gca().add_patch(rect)

    plt.text(
        x_offset + length / 2,
        y_offset + 0.7,
        "log(Element Count)" if log_scale else "Element Count",
        horizontalalignment="center",
        verticalalignment="center",
        fontweight="semibold",
        fontsize=20,
        color="k",
    )

    plt.ylim(-0.15, n_row + 0.1)
    plt.xlim(0.85, n_column + 1.1)

    plt.axis("off")


def hist_elemental_prevalence(formulas, log_scale=False):
    """Plots a histogram of the prevalence of each element in a materials dataset.
    Adapted from https://github.com/kaaiian/ML_figures.

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        log_scale (bool, optional): Whether y-axis is log or linear. Defaults to False.
    """
    plt.figure(figsize=(12, 6))

    elem_tracker, _ = count_elements(formulas)
    non_zero = elem_tracker[elem_tracker != 0].sort_values(ascending=False)

    non_zero.plot.bar(width=0.7, edgecolor="k")

    plt.ylabel = "Element Count" if log_scale else "log(Element Count)"
    if log_scale:
        plt.yscale("log")


def true_vs_pred_with_hist(y_true, y_pred, x_hist=True, y_hist=True):
    plt.figure(figsize=(8, 8))

    left, width = 0.1, 0.65

    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.15]
    rect_histy = [left_h, bottom, 0.15, height]

    ax1 = plt.axes(rect_scatter)
    ax1.tick_params(direction="in", length=7, top=True, right=True)

    ax1.plot(y_true, y_pred, "o", alpha=0.5, label=None, mew=1.2, ms=5.2)

    add_identity(plt.gca(), label="ideal")

    x_range = max(y_true) - min(y_true)
    ax1.set_xlim(max(y_true) - x_range * 1.05, min(y_true) + x_range * 1.05)
    ax1.set_ylim(max(y_true) - x_range * 1.05, min(y_true) + x_range * 1.05)

    ax1.set_ylabel("Predicted value (Units)")
    ax1.set_xlabel("Actual value (Units)")
    ax1.legend(loc=2, framealpha=0.5, handlelength=1.5)

    if x_hist:
        ax2 = plt.axes(rect_histx)
        ax2.hist(y_true, bins=25, rwidth=0.8)
        ax2.axis("off")

    if y_hist:
        ax3 = plt.axes(rect_histy)
        ax3.hist(y_pred, bins=25, orientation="horizontal", rwidth=0.8)
        ax3.axis("off")


def residual(y_true, y_pred):
    fig = plt.figure(figsize=(8, 8))

    y_err = y_pred - y_true

    xmin = np.min(y_true) * 0.9
    xmax = np.max(y_true) / 0.9

    plt.plot(y_true, y_err, "o", alpha=0.5, label=None, mew=1.2, ms=5.2)
    plt.plot([xmin, xmax], [0, 0], "k--", alpha=0.5, label="ideal")

    plt.ylabel("Residual error (Units)")
    plt.xlabel("Actual value (Units)")
    plt.legend(loc="lower right")
    return fig


def residual_hist(y_true, y_pred):
    fig = plt.figure(figsize=(8, 8))

    y_err = y_pred - y_true
    plt.hist(y_err, bins=35, density=True, edgecolor="black")

    kde_true = gaussian_kde(y_err)  # kernel density estimation
    x_range = np.linspace(min(y_err), max(y_err), 100)

    plt.plot(x_range, kde_true(x_range), lw=2, color="red", label="kde")

    plt.xlabel("Residual error (Units)")
    plt.legend(loc=2, framealpha=0.5, handlelength=1)
    plt.show()
    return fig


def show_bar_values(ax, offset=15):
    """Add labels to the end of each bar in a bar chart.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        offset (int): The distance between the labels and the bars.
    """
    for rect in ax.patches:
        y_val = rect.get_height()
        x_val = rect.get_x() + rect.get_width() / 2

        # place label at end of the bar and center horizontally
        ax.annotate(y_val, (x_val, y_val + offset), ha="center")
        # ensure enough vertical space to display label above highest bar
        ax.margins(y=0.1)
