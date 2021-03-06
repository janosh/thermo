"""
This notebook plots the prevalence of different chemical elements in the Gaultois
database in a histogram and onto the periodic table. It also plots histogram for
the four target columns in the Gaultois database: rho, seebeck, kappa, zT.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen import Composition

from thermo.plots import (
    hist_elemental_prevalence,
    ptable_elemental_prevalence,
    show_bar_values,
)
from thermo.utils import ROOT

# %%
cols = ["formula", "rho", "seebeck", "kappa", "zT"]
targets = pd.read_csv(ROOT + "/data/gaultois_targets.csv", header=1)[cols]

DIR = f"{ROOT}/results/target_stats"

# %% [markdown]
# # Elemental Prevalence

# %%
ptable_elemental_prevalence(targets.formula.values, log_scale=True)
plt.savefig(f"{DIR}/ptable_elemental_prevalence.pdf", bbox_inches="tight")


# %%
hist_elemental_prevalence(targets.formula.values, keep_top=20)
plt.savefig(f"{DIR}/hist_elemental_prevalence.pdf", bbox_inches="tight")


# %%
targets["composition"] = [Composition(x) for x in targets.formula]
# Histogram of the number of elements in each composition
x_labels, y_counts = np.unique(
    targets.composition.apply(lambda x: len(x.elements)), return_counts=True
)
ax = plt.bar(x_labels, y_counts, align="center")
plt.xticks(x_labels)
plt.xlabel("number of elements in composition")
plt.ylabel("sample count")
show_bar_values(plt.gca())
plt.savefig(f"{DIR}/hist_number_of_elements_in_composition.pdf", bbox_inches="tight")


# %%
xlabels = [
    "Electrical Resistivity [Ohm m]",
    "Seebeck Coefficient [V/K]",
    "Thermal Conductivity [W/(m K)]",
    "Figure of Merit",
]
axs = targets.hist(bins=50, figsize=[15, 3], layout=[1, 4])
for ax, name in zip(axs.ravel(), xlabels):
    ax.set_xlabel(name)

plt.savefig(f"{DIR}/4_targets_hist.pdf", bbox_inches="tight")
