import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from matplotlib.cm import YlGn
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pymatgen.core import Composition
from scipy.stats import pearsonr, spearmanr

from utils import ROOT

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


def err_decay(err_name, *args, title=None, bare=False):
    """Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error

    Args:
        err_name (str): usually MAE or MSE
        title (str, optional): plot title
        bare (bool, optional): whether to include title and axis labels
    """
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

    plt.show()


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

    plt.show()


def true_vs_pred(y_test, y_pred, y_std=None, title=None, bare=False):
    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    plt.errorbar(y_test, y_pred, yerr=y_std, **styles)
    if not bare:
        plt.xlabel("$y_\\mathrm{test}$")
        plt.ylabel("$y_\\mathrm{pred}$")
        plt.title(title)
    add_identity(plt.gca())
    prefix = f"$\\epsilon_\\mathrm{{mae}} = {np.abs(y_test - y_pred).mean():.3g}$\n"
    display_corr(y_test, y_pred, x=0.15, halign="left", prefix=prefix)

    plt.show()


def std_vs_pred(y_std, y_pred, title=None):
    plt.scatter(y_std, y_pred)
    plt.xlabel("$y_\\mathrm{std}$")
    plt.ylabel("$y_\\mathrm{pred}$")
    plt.title(title)

    plt.show()


def loss_history(loss_histories):
    for name, loss in loss_histories.items():
        plt.plot(loss, label=name)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

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

    plt.show()


def cv_mse_decay(df, title=None, bare=False):
    ax = sns.lineplot(data=df[["err_decay_by_std", "true_err_decay"]])
    display_err_decay_avr_dist(df.err_decay_by_std, df.true_err_decay)
    if title:
        plt.title(title)
    if bare:
        ax.get_legend().remove()
        plt.xlabel("")
        plt.ylabel("")

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
