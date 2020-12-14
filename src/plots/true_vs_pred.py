import numpy as np
from matplotlib import pyplot as plt

from .utils import add_identity, add_text_box, corr_text


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


def scatter_with_hist(xs, ys, ax=None, bins=100, xlabel=None, ylabel=None, **kwargs):
    plt.figure(figsize=(8, 8))

    bottom, height = left, width = 0.1, 0.65

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, left + width, width, 0.1]
    rect_histy = [left + width, bottom, 0.1, height]

    ax_scatter = plt.axes(ax or rect_scatter)
    ax_scatter.plot(xs, ys, "o", alpha=0.5, **kwargs)

    add_identity(ax, label="ideal")

    ax.set_ylabel(ylabel or "predicted")
    ax.set_xlabel(xlabel or "actual")
    ax.legend(loc=2, frameon=False)

    # x_hist
    ax_histx = plt.axes(rect_histx)
    ax_histx.hist(xs, bins=bins, rwidth=0.8)
    ax_histx.axis("off")

    # y_hist
    ax_histy = plt.axes(rect_histy)
    ax_histy.hist(ys, bins=bins, orientation="horizontal", rwidth=0.8)
    ax_histy.axis("off")
