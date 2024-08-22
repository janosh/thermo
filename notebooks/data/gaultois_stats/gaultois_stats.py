"""This notebook plots the prevalence of different chemical elements in the Gaultois
database in a histogram and onto the periodic table. It also plots histogram for
the four target columns in the Gaultois database: rho, seebeck, kappa, zT.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatviz as pmv
from pymatgen.core import Composition

from thermo.utils import ROOT


# %%
cols = ["formula", "rho", "seebeck", "kappa", "zT"]
targets = pd.read_csv(ROOT + "/data/gaultois_targets.csv", header=1)[cols]


# %% [markdown]
# # Elemental Prevalence


# %%
ax = pmv.ptable_heatmap(targets.formula.values, log_scale=True)
pmv.save_fig(ax, "gaultois_elements.pdf", bbox_inches="tight")


# %%
ax = pmv.elements_hist(targets.formula.values, keep_top=20, voffset=20)
pmv.save_fig(ax, "hist_elements.pdf", bbox_inches="tight")


# %%
targets["composition"] = [Composition(x) for x in targets.formula]
# Histogram of the number of elements in each composition
x_labels, y_counts = np.unique(
    targets.composition.map(lambda x: len(x.elements)), return_counts=True
)
ax = plt.bar(x_labels, y_counts, align="center")
plt.xticks(x_labels)
ax.set(xlabel="number of elements in composition", ylabel="sample count")
pmv.powerups.annotate_bars(ax)
pmv.save_fig(ax, "hist_number_of_elements_in_composition.pdf", bbox_inches="tight")


# %% [markdown]
# # Target Histograms


# %%
x_labels = [
    "Electrical Resistivity [Ohm m]",
    "Seebeck Coefficient [V/K]",
    "Thermal Conductivity [W/(m K)]",
    "Figure of Merit",
]
axs = targets.hist(bins=50, figsize=[15, 3], layout=[1, 4])
for ax, name in zip(axs.ravel(), x_labels):
    ax.set_xlabel(name)

pmv.save_fig(ax, "4_targets_hist.pdf", bbox_inches="tight")
