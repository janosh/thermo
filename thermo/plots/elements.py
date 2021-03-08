import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import YlGn
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pymatgen import Composition

from thermo.utils import ROOT


def count_elements(formulas):
    """Count occurrences of each chemical element in a materials dataset.

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]

    Returns:
        pd.Series: Number of appearances for each element in formulas.
    """

    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")
    elem_counts = pd.Series(0, index=ptable.symbol)  # symbols = [H, He, Li, etc.]

    for formula in formulas:
        formula_dict = Composition(formula).as_dict()
        elem_count = pd.Series(formula_dict, name="count")
        elem_counts = elem_counts.add(elem_count, fill_value=0)

    return elem_counts


def ptable_elemental_prevalence(formulas=None, elem_counts=None, log_scale=False):
    """Colormap the periodic table according to the prevalence of each element
    in a materials dataset.
    Adapted from https://github.com/kaaiian/ML_figures.
    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        log_scale (bool, optional): Whether color map scale is log or linear.
    """
    if (formulas is None and elem_counts is None) or (
        formulas is not None and elem_counts is not None
    ):
        raise ValueError("provide either formulas or elem_counts, not neither nor both")

    if formulas is not None:
        elem_counts = count_elements(formulas)

    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")

    n_row = ptable.row.max()
    n_column = ptable.column.max()

    plt.figure(figsize=(n_column, n_row))

    rw = rh = 0.9  # rectangle width/height
    count_min = elem_counts.min()
    count_max = elem_counts.max()

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
        count = elem_counts[symbol]
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


def hist_elemental_prevalence(formulas, log_scale=False, keep_top=None):
    """Plots a histogram of the prevalence of each element in a materials dataset.
    Adapted from https://github.com/kaaiian/ML_figures.

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        log_scale (bool, optional): Whether y-axis is log or linear. Defaults to False.
    """
    # plt.figure(figsize=(14, 7))
    # plt.rcParams.update({"font.size": 18})

    elem_counts = count_elements(formulas)
    non_zero = elem_counts[elem_counts != 0].sort_values(ascending=False)
    if keep_top is not None:
        non_zero = non_zero.head(keep_top)
        plt.title(f"top {keep_top} elements")

    non_zero.plot.bar(width=0.7, edgecolor="black")

    plt.ylabel("log(Element Count)" if log_scale else "Element Count")
    if log_scale:
        plt.yscale("log")
