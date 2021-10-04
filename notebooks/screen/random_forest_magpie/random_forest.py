"""
This notebook screens a combined list of synthesizable materials from ICSD and
COD databases for promising thermoelectric candidates using random forest
regression.
"""


# %%
import os
import pickle

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
from ml_matrics import marchenko_pastur, ptable_heatmap, qq_gaussian
from sklearn.model_selection import train_test_split

from thermo.correlation import expected_rand_obj_val, rand_obj_val_avr
from thermo.data import dropna, load_gaultois, load_screen
from thermo.plots import plot_output
from thermo.rf import rf_predict
from thermo.utils import ROOT


# %%
gaultois_magpie_feas, gaultois_targets = load_gaultois()
gaultois_targets, gaultois_magpie_feas = dropna(gaultois_targets, gaultois_magpie_feas)
zT = gaultois_targets.zT

# magpie_screen are the Magpie features for the screening set without temperature yet
screen_ids, magpie_screen = load_screen()


# %%
# Form Cartesian product between screen features and the 4 temperatures ([300, 400, 700,
# 1000] Kelvin) found in Gaultois' database. We'll predict each material at all 4 temps.
# Note: None of the composition are predicted to achieve high zT at 300, 400 Kelvin.
# Remove those to save time.
temps = (700, 1000)  # Kelvin
temp_col = np.tile(temps, len(magpie_screen))

screen_features = magpie_screen.loc[magpie_screen.index.repeat(len(temps))]
screen_features.insert(0, "T", temp_col)

candidates = screen_ids.loc[screen_ids.index.repeat(len(temps))]
candidates["T"] = temp_col
candidates.set_index(["id", "T"], inplace=True, append=True)


# %% Validate good performance with a train-test split before screening
X_train, X_test, zT_train, zT_test = train_test_split(
    gaultois_magpie_feas, zT, test_size=0.1
)

n_trees = 300
zT_test_pred, zT_test_std, _ = rf_predict(
    X_train, zT_train, X_test, n_estimators=n_trees, verbose=1
)

# random forest is slightly underconfident
qq_gaussian(zT_test.values, zT_test_pred, 0.9 * zT_test_std)
plt.show()

plot_output(zT_test.values, zT_test_pred, 0.9 * zT_test_std)


# %% [markdown]
# # Screening


# %%
zT_pred, zT_std, forest = rf_predict(
    gaultois_magpie_feas, zT, screen_features, n_estimators=n_trees, verbose=1
)


with open(f"forest-{n_trees=}.pkl", "wb") as file:
    pickle.dump(forest, file)


# %%
# candidates["zT_pred"] = zT_pred
# candidates["zT_std"] = zT_std

# candidates.to_csv("candidates.csv")
candidates = pd.read_csv(f"candidates-{n_trees=}.csv", index_col=["index", "id", "T"])
# with open(f"forest-{n_trees=}.pkl", "rb") as file:
#    forest = pickle.load(file)


# %% [markdown]
# # Coarse triaging


# %%
min_zT_pred, max_zT_std = 0.5, 0.33

for temp, group in candidates.reset_index().groupby("T"):
    plt.scatter(x=group.zT_std, y=group.zT_pred, label=f"{temp} K", s=5)
plt.legend(title="Temperature", markerscale=3)
plt.title("Uncertainty vs predicted zT for each temperature")
plt.xlabel("zT_std")
plt.ylabel("zT_pred")
plt.axvspan(xmin=0, xmax=max_zT_std, ymin=min_zT_pred, alpha=0.15)
plt.xlim(0, None)
plt.savefig("zT_pred-vs-zT_std.png", bbox_inches="tight", dpi=200)


# %%
lrhr_idx = np.logical_and(zT_std < max_zT_std, zT_pred > min_zT_pred)
(lrhr_candidates := candidates[lrhr_idx])

lrhr_candidates.to_csv("lrhr_candidates.csv", float_format="%g")
# lrhr_candidates = pd.read_csv("lrhr_candidates.csv", index_col=[0, "id", "T"])


# %%
# Save materials predicted to have highest zT with no concern for estimated
# uncertainty to see if uncertainty estimation reduces the false positive rate.
# greedy_candidates = candidates.sort_values("zT_pred", ascending=False)[:1000]
# greedy_candidates.to_csv("greedy_candidates.csv", float_format="%g")
greedy_candidates = pd.read_csv("greedy_candidates.csv", index_col=[0, "id", "T"])


# %%
lrhr_candidates.plot.scatter(x="zT_std", y="zT_pred")
plt.savefig("lrhr_materials_pred_vs_std.pdf", bbox_inches="tight")
r_p = lrhr_candidates[["zT_std", "zT_pred"]].corr().iloc[0, 1]
plt.text(0.03, 0.05, f"${r_p = :.3f}$", transform=plt.gca().transAxes)


# %% [markdown]
# # Compute correlations between low-risk high-return materials


# %%
zT_corr = forest.get_corr(
    screen_features.iloc[lrhr_candidates.index.get_level_values(0)]
)
zT_corr = pd.DataFrame(
    zT_corr, columns=lrhr_candidates.formula, index=lrhr_candidates.index
)
zT_corr.set_index(zT_corr.columns, append=True, inplace=True)
zT_corr.to_csv("correlation_matrix.csv", float_format="%g")
# zT_corr = pd.read_csv("correlation_matrix.csv", index_col=[0, "id", "T", "formula"])


# %%
color_ax = plt.matshow(zT_corr)
plt.colorbar(color_ax, fraction=0.047, pad=0.02)
plt.gcf().set_size_inches(12, 12)

plt.savefig("correlation_matrix_rf.pdf", bbox_inches="tight")


# %%
# the weakly correlated elements contain lots of arsenide which is absent from
# the gaultois training set while the strongly correlated materials contain
# zero arsenide (compare these plots with notebooks/data/gaultois_elements.pdf)
ptable_heatmap(zT_corr.columns[:190])
plt.title("elements in weakly correlated (blue) part of zT correlation matrix")
plt.savefig("zT_corr-elements-cols-0-190.pdf")

ptable_heatmap(zT_corr.columns[190:])
plt.title("elements in strongly correlated (yellow) part of zT correlation matrix")
plt.savefig("zT_corr-elements-cols-190-end.pdf")


# %%
n_candidates = len(lrhr_candidates)
ptable_heatmap(lrhr_candidates.formula)
plt.title(f"elemental prevalence among {n_candidates} low-risk high-return candidates")
plt.savefig("lrhr-ptable-elements.pdf")

ptable_heatmap(greedy_candidates.head(n_candidates).formula)
plt.title(f"elemental prevalence among {n_candidates} greedy candidates")
plt.savefig("greedy-ptable-elements.pdf")


# %%
# https://www.pnas.org/content/113/48/13564
# Create the correlation matrix and find the eigenvalues
N = len(zT_corr)
p = forest.n_estimators

marchenko_pastur(zT_corr, gamma=p / N)
plt.title(
    "Marchenko-Pastur distribution of the MNF zT correlation matrix\n"
    f"{p = }, {N = }, gamma = p / N = {p / N:.2f}"
)
plt.yscale("log")
plt.savefig("marchenko-pastur-dist.png")


# %%
# Eigenvalues larger than the largest theoretical eigenvalue
# the std dev sigma = 1 here so we don't write it explicitly
max_theoretical_eval = (1 + np.sqrt(N / p)) ** 2

evals, evecs = np.linalg.eigh(zT_corr)

print(evals[evals > max_theoretical_eval])


# %% [markdown]
# # Fine Triaging
# Helpful links for the discrete constrained optimization problem of
# finding the p least correlated materials out of n predictions:
# - [Find k of n items with least pairwise correlations](
# https://stats.stackexchange.com/q/73125)
# - [Least correlated subset of random variables from a correlation matrix](
# https://stats.stackexchange.com/q/110426)


# %%
# The idea for this way of reducing correlation came from
# https://stats.stackexchange.com/a/327822. Taking the element-wise
# absolute value (rather than squaring) and then summing gives similar results.
least_total_corr_candidates = lrhr_candidates.copy(deep=True)

least_total_corr_candidates["rough_correlation"] = (zT_corr ** 2).sum().values

least_total_corr_candidates = (
    least_total_corr_candidates.reset_index()
    .rename(columns={"index": "orig_index"})
    .sort_values(by="rough_correlation")
)

least_total_corr_candidates.to_csv(
    "least_total_corr_candidates.csv", index=False, float_format="%g"
)


# %%
# Set environment variable GRB_LICENSE_FILE so that Gurobi finds its license file.
# An academic license can be obtained for free at
# https://gurobi.com/downloads/end-user-license-agreement-academic.
os.environ["GRB_LICENSE_FILE"] = f"{ROOT}/hpc/gurobi.lic"
# Create a model for solving the quadratic optimization problem of selecting p out of n
# materials with least pairwise correlation according to the correlation matrix zT_corr.
grb_model = Model("quadratic_problem")
grb_model.params.LogFile = "gurobi.log"


# %%
# Create decision variables.
dec_vars = grb_model.addVars(len(lrhr_candidates), vtype=GRB.BINARY).values()


# %%
# Define the model objective to minimize the sum of pairwise correlations.
obj = zT_corr.dot(dec_vars).dot(dec_vars)
grb_model.setObjective(obj, GRB.MINIMIZE)


# %%
# Add L1 constraint on dec_vars so that the optimization returns at least
# n_select formulas. If the model finds more at equal correlation, even better.
n_select = 200
constr = grb_model.addConstr(quicksum(dec_vars) >= n_select, "l1_norm")


# %%
grb_model.optimize()


# %%
# Save selected materials to dataframe and CSV file.
dec_vals = [bool(var.x) for var in dec_vars]
print(f"final objective value: {zT_corr.dot(dec_vals).dot(dec_vals) = :.3f}")
gurobi_candidates = lrhr_candidates.iloc[dec_vals]
# gurobi_candidates.to_csv("gurobi_candidates.csv")


# %%
ptable_heatmap(gurobi_candidates.formula)
plt.title(f"elements in {len(gurobi_candidates)} Gurobi candidates")
plt.savefig("gurobi-ptable-elements.pdf")


# %%
for name, df in zip(
    ["gurobi_candidates", "least_total_corr_candidates"],
    [gurobi_candidates, least_total_corr_candidates.iloc[:n_select]],
):
    df.sort_values(["formula", "T"]).reset_index().to_latex(
        f"{name}.tex",
        columns=["formula", "database", "id", "T", "zT_pred", "zT_std"],
        float_format="%.3g",
        index=False,
    )


# %% [markdown]
# # Comparing greedy and Gurobi solution


# %%
# least_total_corr_candidates contains all low-risk high-return materials sorted by
# sum of squared correlations with all other materials. If either greedy or Gurobi
# (or both) picked materials entirely at random, we would expect formulas chosen by
# Gurobi to have an average index in the list equal to the total list's average index.
# The degree to which the average index of Gurobi materials in the greedy list is lower
# than average list index is an indicator of agreement between the two methods.
gurobi_in_greedy = least_total_corr_candidates.orig_index.isin(gurobi_candidates.index)
greedy_avg_index = (
    least_total_corr_candidates[gurobi_in_greedy].index.to_series().mean()
)

print(
    "Average index of materials chosen by Gurobi in the list\n"
    f"sorted according to least squared correlation: {greedy_avg_index}\n"
    f"vs the average index of the total list: {(len(lrhr_candidates) + 1) / 2}"
)


# %%
greedy_indices_in_corr_mat = lrhr_candidates.index.isin(
    least_total_corr_candidates.orig_index[:n_select]
)
greedy_obj_val = zT_corr.values.dot(greedy_indices_in_corr_mat).dot(
    greedy_indices_in_corr_mat
)

avr_rand_obj_val = rand_obj_val_avr(zT_corr, n_select, (n_repeats := 50))

# If len(zT_corr) >> 500, expected_rand_obj_val will take a long time due to cubic
# scaling. Consider decreasing max_risk or increasing min_ret in
# filter_low_risk_high_ret to decrease len(zT_corr).
exp_rand_obj_val = expected_rand_obj_val(zT_corr, n_select)

print(
    f"objective values:\n- Gurobi: {grb_model.objVal:.4g}\n"
    f"- greedy: {greedy_obj_val:.4g}\n"
    f"- average of {n_repeats} random draws: {avr_rand_obj_val:.4g}\n"
    f"- expectation value of random solution: {exp_rand_obj_val:.4g}"
)


# %% [markdown]
# # Correlation between low-risk high-return materials


# %%
dft_seebeck = pd.read_csv("dft/gurobi_seebeck.csv")[
    ["formula", "300", "400", "700", "1000"]
].set_index("formula")


# %%
rf_seebeck = (
    screen_features[["formula", "seebeck_abs_pred", "T"]]
    .loc[screen_features.formula.isin(dft_seebeck.index)]
    .pivot(index="formula", columns="T", values="seebeck_abs_pred")
) * 1e6  # conversion from SI to common units (V/K -> uV/K)

# rf_seebeck.seebeck_abs_pred = rf_seebeck.seebeck_abs_pred * 1e6


# %%
dft_seebeck.columns = rf_seebeck.columns
methods = ["pearson", "spearman"]
dft_rf_seebeck_abs_corr = pd.concat(
    [rf_seebeck.corrwith(abs(dft_seebeck), axis=1, method=m) for m in methods],
    axis=1,
    keys=methods,
)
dft_rf_seebeck_abs_corr.mean()


# %%
seebeck_abs_preds_and_corrs = pd.concat(
    [dft_rf_seebeck_abs_corr, rf_seebeck, abs(dft_seebeck)],
    axis=1,
    keys=["correlation", "RF", "DFT"],
)
seebeck_abs_preds_and_corrs.loc["mean"] = seebeck_abs_preds_and_corrs.mean()
seebeck_abs_preds_and_corrs.to_latex(
    "rf_seebeck_abs_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)
