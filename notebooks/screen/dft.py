"""This notebook plots DFT results for thermoelectric properties of several
candidate materials identified via random forest regression and portfolio-like
risk management. See src/notebooks/screen/random_forest.py for details.
"""

# %%
import pandas as pd
import pymatviz as pmv
from matplotlib import pyplot as plt

from thermo.utils import ROOT


OUT_DIR = ROOT + "/results/screen/dft"


# %%
# convert excel file into usable format
zT_el_greedy_gurobi = pd.read_excel(f"{OUT_DIR}/zT_el_greedy&gurobi.xlsx").dropna()
zT_el_greedy_gurobi.columns = range(len(zT_el_greedy_gurobi.columns))


# %% Get mask to distinguish string values in column 0.
m1 = pd.to_numeric(zT_el_greedy_gurobi[0], errors="coerce").isna()

# Create new column 0 filled with strings.
zT_el_greedy_gurobi.insert(0, "tmp", zT_el_greedy_gurobi[0].where(m1).ffill())

# Mask for values that are not the same in both columns.
m2 = zT_el_greedy_gurobi["tmp"].ne(zT_el_greedy_gurobi[0])

# Create MultiIndex.
zT_el_greedy_gurobi = zT_el_greedy_gurobi.set_index(["tmp", 0])


# %% Assign new column names by first row.
zT_el_greedy_gurobi.columns = [
    int(float(str(c).replace("K", ""))) for c in zT_el_greedy_gurobi.iloc[0]
]

# Filter out by mask and remove index, columns names.
zT_el_greedy_gurobi = (
    zT_el_greedy_gurobi[m2.values]
    .rename_axis(["formula", "n"])
    .rename_axis(None, axis=1)
)


# %%
zT_el_greedy_gurobi.to_csv(f"{OUT_DIR}/zT_el_greedy&gurobi.csv", float_format="%g")

#
# # %%
# zT_el_greedy_gurobi = pd.read_csv(f"{OUT_DIR}/zT_el_greedy&gurobi.csv").set_index(
#     ["formula", "n"]
# )


# %%
fig, axs = plt.subplots(5, 4, figsize=(15, 15))
fig.tight_layout()
for ax, (key, material) in zip(axs.flat, zT_el_greedy_gurobi.groupby("formula")):
    material.T.plot(ax=ax)
    ax.get_legend().remove()
    ax.set_title(key)

plt.subplots_adjust(hspace=0.2, wspace=0.3)  # increase height & width between subplots


# %%
GePtSe, Bi2Te3 = (zT_el_greedy_gurobi.loc[m1] for m1 in ["GePtSe", "Bi2Te3"])


# %%
for name, df in zip(["GePtSe", "Bi2Te3"], [GePtSe, Bi2Te3]):
    ax = df.T.plot(marker="o")
    if name != "GePtSe":
        ax.get_legend().remove()
    else:
        ax.legend([f"$10^{{{i}}}$" for i in range(17, 22)])
    pmv.save_fig(
        ax, f"{OUT_DIR}zT_el_{name}.pdf", bbox_inches="tight", transparent=True
    )


# %% [markdown]
# ## GePtSe vs Bi2Te3: $zT$


# %%
zT_GePtSe = pd.read_csv(f"{OUT_DIR}/zT_GePtSe.csv", index_col="T (K)")
zT_GePtSe.columns = pd.to_numeric(zT_GePtSe.columns, errors="ignore")
zT_GePtSe.index = [f"$10^{{{idx.split('^')[-1]}}}$" for idx in zT_GePtSe.index]
zT_Bi2Te3 = pd.read_csv(f"{OUT_DIR}/zT_Bi2Te3.csv", comment="#")


# %%
ax = zT_GePtSe.T.plot(marker="o")
zT_Bi2Te3.plot(x="T (K)", y="zT", marker="o", label="Bi$_2$Te$_3$", ax=ax)
# Reverse figure legend to match up with order of lines.
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
# ax.set_ylabel("zT")
ax.set(xlabel="", ylabel="")
pmv.save_fig(
    ax, f"{OUT_DIR}/zT_GePtSe_vs_Bi2Te3.pdf", bbox_inches="tight", transparent=True
)


# %% [markdown]
# ## GePtSe vs Bi2Te3: $\kappa_\text{ph}$


# %%
kph_GePtSe = pd.read_csv(f"{OUT_DIR}/kappa_ph_GePtSe.csv", comment="#")
kph_Bi2Te3 = pd.read_csv(f"{OUT_DIR}/kappa_ph_Bi2Te3.csv", comment="#")


# %%
ax = kph_GePtSe.plot(marker="o", x="T")
kph_Bi2Te3.plot(marker="o", x="T", ax=ax)
plt.legend(["GePtSe", "Bi$_2$Te$_3$"])
ax.set_ylabel(r"$\kappa_\mathrm{ph}$")
ax.set_xlabel("T (K)")
pdf_path = f"{OUT_DIR}/kappa_ph_GePtSe_vs_Bi2Te3.pdf"
pmv.save_fig(ax, pdf_path, bbox_inches="tight", transparent=True)
